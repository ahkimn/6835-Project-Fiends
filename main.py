"""
Main algorithm for extracting finger position from Kinect v2 RGBD data
"""
import argparse
import load_data
import os
import cv2
import numpy as np
from numpy.linalg import svd
from matplotlib import pyplot as plt

from scipy.ndimage.filters import gaussian_filter

# Directory containing data sequences
# SEQUENCE_DIRECTORY = './data/sequences'
# SEQUENCE_DIRECTORY = '../rgbd_scan/rgbd_scan/data'
SEQUENCE_DIRECTORY = '../rgbd_scan/rgbd_scan/rgbd_scan'

# Thresholds on HSV hue for selecting RoI
HUE_HIGH = 15
HUE_LOW = 5

# Minimum size of reasons of uniform hue to be used as candidate
#   RoI
MIN_PLANE = 10000

# Number of initial frames to sample to determine location of
#   estimated plane
N_CALIBRATION_FRAMES = 10

# Min max thresholds for depth distance to estimated plane
#   for suspected finger locations
DEPTH_NOISE_THRESHOLD = 15
DEPTH_FINGER_THRESHOLD = 130
GRADIENT_THRESHOLD = 30

#
MIN_FINGER_AREA = 1000
MAX_FINGER_AREA = 50000
MIN_CIRCULARITY = 0.2
MAX_CIRCULARITY = 0.8
MIN_ASPECT_RATIO = 0.3
MIN_EXTENT = 0.3

# RandomState for deterministic sampling of points in
#   plane estimation
RS = np.random.RandomState(seed=0)

# Constants for size of interface (used in mapping from)
#   estimated plane to interface
INTERFACE_HEIGHT = 100
INTERFACE_WIDTH = 100

# Upscale factor of joint data relative to image data
JOINT_SCALE = 1

# Joint data horizontal flip relative to image data
JOINT_FLIP = False


def threshold_img(img):
    """
    Convert RGB to HSV image, and threshold on Hue to select RoI

    Args:
        img (np.ndarray): RGB image to rhreshold

    Returns:
        np.ndarray: Description
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    hue = hsv[:, :, 0][:]
    hue = gaussian_filter(hue, sigma=7)

    # Filter Hue
    mask = np.logical_and(hue < HUE_HIGH, hue >= HUE_LOW)
    hue[mask] = 255
    hue[~mask] = 0

    max_coverage = 0

    best_contour = None

    __, contours, __ = cv2.findContours(hue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("hue", hue)

    # Iterate through contours; select one with largest area
    for c in contours:
        rect = cv2.boundingRect(c)

        for const in np.arange(0.05, 0.15, 0.01):
            epsilon = const * cv2.arcLength(c, True)
            # Simplify contour using polynomial approximation
            approx = cv2.approxPolyDP(c, epsilon, True)

            if len(approx) == 4:
                break

        # img2 = cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
        # cv2.imshow('ex_contour', img2)
        # cv2.waitKey(0)

        rect = cv2.boundingRect(approx)
        x, y, w, h = rect

        # Determine if region of contour is large
        if w * h > MIN_PLANE:

            mask = np.zeros(hue.shape, np.uint8)
            mask[y:y + h, x:x + w] = hue[y:y + h, x:x + w]

            coverage = np.sum(mask)

            if coverage > max_coverage:
                best_contour = approx
                max_coverage = coverage

    return best_contour


def get_bounds(rgb_list, n_frames):
    """
    Determine RoI bounds over n_frames

    Args:
        rgb_list (np.ndarray): RGB image data
        n_frames (TYPE): Number of frames to use to determine RoI
            contour

    Returns:
        (np.ndarray): Best approximate bounds of RoI
    """
    n_frames = min(len(rgb_list), n_frames)

    valid_contours = []

    # Select best RoI contour of length 4 for each frame
    for i in range(n_frames):
        c = threshold_img(rgb_list[i])
        if len(c) == 4:
            valid_contours.append(c)

    # print(valid_contours)

    # Use median contour values across frames as best approximation for RoI
    median_c = np.array([
        int(np.median([c.flatten()[i] for c in valid_contours]))
        for i in range(8)]).reshape(4, 1, 2)

    return median_c


def get_ordered_bounds(bounds):
    """
    Reorder bounds of RoI so that the four points are in order:
        top left, top right, bottom right, bottom left

    Args:
        bounds (np.ndarray): Original bounds of RoI

    Returns:
        (tuple): Tuple of each point (y, x) of reordered bounds of RoI
    """
    flat = bounds.reshape(4, 2)
    avg_x = np.mean(flat[:, 1])
    avg_y = np.mean(flat[:, 0])

    p_1, p_2, p_3, p_4 = None, None, None, None

    for i in range(4):

        if flat[i][0] < avg_y and flat[i][1] < avg_x:

            p_1 = flat[i][:]

        elif flat[i][0] < avg_y and flat[i][1] >= avg_x:

            p_2 = flat[i][:]

        elif flat[i][0] >= avg_y and flat[i][1] >= avg_x:

            p_3 = flat[i][:]

        else:

            p_4 = flat[i][:]

    assert(all(p is not None for p in [p_1, p_2, p_3, p_4]))
    return p_1, p_2, p_3, p_4


def get_transform_matrix(bounds):
    """
    Determine affine transformation from RoI bounds to interface rectangle

    Args:
        bounds (np.ndarray): Bounds of RoI

    Returns:
        (np.ndarray): 3x3 matrix reflecting affine transformation
    """
    bound_rect = cv2.boundingRect(bounds)
    x, y, w, h = bound_rect

    p_1, p_2, p_3, p_4 = get_ordered_bounds(bounds)

    # Assume interface coordinates begin at (0, 0) at upper left
    # upper_left
    X_1, Y_1 = 0, 0
    x_1, y_1 = p_1
    # upper right
    X_2, Y_2 = INTERFACE_WIDTH, 0
    x_2, y_2 = p_2
    # lower right
    X_3, Y_3 = INTERFACE_WIDTH, INTERFACE_HEIGHT
    x_3, y_3 = p_3
    # lower left
    X_4, Y_4 = 0, INTERFACE_HEIGHT
    x_4, y_4 = p_4

    M = np.array(
        [[x_1, y_1, 1, 0, 0, 0, -X_1 * x_1, -X_1 * y_1, 0],
         [0, 0, 0, x_1, y_1, 1, -Y_1 * x_1, -Y_1 * y_1, 0],
         [x_2, y_2, 1, 0, 0, 0, -X_2 * x_2, -X_2 * y_2, 0],
         [0, 0, 0, x_2, y_2, 1, -Y_2 * x_2, -Y_2 * y_2, 0],
         [x_3, y_3, 1, 0, 0, 0, -X_3 * x_3, -X_3 * y_3, 0],
         [0, 0, 0, x_3, y_3, 1, -Y_3 * x_3, -Y_3 * y_3, 0],
         [x_4, y_4, 1, 0, 0, 0, -X_4 * x_4, -X_4 * y_4, 0],
         [0, 0, 0, x_4, y_4, 1, -Y_4 * x_4, -Y_4 * y_4, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    b = np.array([X_1, Y_1, X_2, Y_2, X_3, Y_3, X_4, Y_4, 1])
    T = np.linalg.solve(M, b).reshape(3, 3)
    return T


def transform_coords(T, coords):
    """
    Transform image to UI coords, given the affine transformation
        between the two

    Args:
        T (np.ndarray): Affine transformation from transform iamge -> UI coords
        coords (np.ndarray): image coordinates to transform (x, y)

    Returns:
        (np.ndarray): UI coordinates if within bounds; else None is returned
    """
    print(coords)

    img_coords = [coords[0], coords[1], 1]
    ui_coords = np.dot(T, img_coords)
    ui_coords /= ui_coords[-1]

    if ui_coords[0] > INTERFACE_WIDTH or ui_coords[0] < 0:
        return None
    if ui_coords[1] > INTERFACE_HEIGHT or ui_coords[1] < 0:
        return None

    return ui_coords[:2]


def est_plane_equation(depth, bounds, n_points=40):
    """
    Estimate plane equation (Depth in terms of image (x, y))

    Args:
        depth (np.ndarray): Depth image
        bounds (np.ndarray): Bounds of RoI
        n_points (int, optional): Number of points to sample in RoI
            to obtain estimated plane

    Returns:
        (function): Function to calculate depth of estimated plane at
            each image (x, y)
    """
    points = np.zeros((n_points, 3))
    height, width = depth.shape

    valid = 0

    # Sample points within RoI
    while valid < n_points:

        x, y = RS.randint(width), RS.randint(height)

        if cv2.pointPolygonTest(bounds, (x, y), True) > 5:

            points[valid] = [x, y, depth[y, x]]
            valid += 1

    # Estimate plane from sampled points
    p, N = planeFit(points)

    def est_depth(x, y):
        """
        Function to obtain predicted depth given image (x, y)

        Args:
            x (float): x image coordinate
            y (float): y image coordinate

        Returns:
            float: Depth value of plane at (x, y)
        """
        z = (N[0] * (x - p[0]) + N[1] * (y - p[1]) - (p[2] * N[2]))
        z /= N[2]
        return -z

    return est_depth


def plt_depth_diff(depth, est_depth):
    """
    Plot depth deviation from estimate plane of RoI over entire image

    Args:
        depth (np.ndarray): Depth image
        est_depth (function): Function to calculate depth vs. image coordinate
    """
    tmp = np.zeros(depth.shape)

    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            tmp[i, j] = est_depth(j, i) - depth[i, j]

    plt.figure(1)
    plt.imshow(-np.abs(tmp))
    plt.axis('off')
    plt.colorbar(fraction=0.026, pad=0.04)
    plt.title('Absolute Depth Difference to Estimated Plane', fontsize=18)

    plt.figure(2)
    plt.imshow(depth)
    plt.axis('off')
    plt.colorbar(fraction=0.026, pad=0.04)
    plt.title('Absolute Depth Difference to Estimated Plane', fontsize=18)

    plt.show()

def plt_depth_diff_map(depth, est_depth):
    """
    Plot depth deviation from estimate plane of RoI over entire image

    Args:
        depth (np.ndarray): Depth image
        est_depth (function): Function to calculate depth vs. image coordinate
    """
    tmp = np.zeros(depth.shape)
    thresh = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint8)
    thresh2 = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint8)

    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            tmp_val = est_depth[i, j] - depth[i, j]
            if tmp_val > 30 and tmp_val < 100:
                tmp[i, j] = tmp_val
                thresh[i, j] = 255
            elif tmp_val > 0 and tmp_val < 25:
                thresh2[i, j] = 255

    gradients_float = cv2.Laplacian(depth, cv2.CV_64F, ksize=3)
    gradients_float_abs = gradients_float.clip(min=0)
    # gradients_float_abs = np.absolute(gradients_float)
    gradients = np.uint8(gradients_float_abs)

    ret, gradients_bin = cv2.threshold(gradients,30,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    gradients_bin = cv2.morphologyEx(gradients_bin, cv2.MORPH_OPEN, kernel)
    # gradients_bin = cv2.erode(gradients_bin, kernel,iterations = 3)
    mask = cv2.bitwise_and(thresh, gradients_bin)

    kernel = np.ones((5,5),np.uint8)
    # opening = cv2.dilate(cv2.erode(mask, kernel, 2), kernel, 2)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    roi = cv2.bitwise_and(thresh, opening)
    roi_depth = cv2.bitwise_and(depth, depth, mask=opening)
    gradients_float = cv2.Laplacian(roi_depth, cv2.CV_64F, ksize=3)
    gradients_float_abs = np.absolute(gradients_float)
    gradients_roi = np.uint8(gradients_float_abs)

    cv2.imshow("thresh", cv2.resize(thresh, (int(width/2.), int(height/2.))))
    cv2.imshow("gradient", cv2.resize(gradients, (int(width/2.), int(height/2.))))
    cv2.imshow("binary gradient", cv2.resize(gradients_bin, (int(width/2.), int(height/2.))))
    cv2.imshow("mask", cv2.resize(mask, (int(width/2.), int(height/2.))))
    cv2.imshow("opening", cv2.resize(opening, (int(width/2.), int(height/2.))))
    # cv2.imshow("roi", opening)
    cv2.imshow("gradients roi", cv2.resize(gradients_roi, (int(width/2.), int(height/2.))))
    cv2.waitKey()
    cv2.destroyAllWindows()


def depth_gradient(depth):
    gradients_float = cv2.Laplacian(depth, cv2.CV_64F, ksize=3)
    gradients_float_abs = gradients_float.clip(min=0)
    gradients = np.uint8(gradients_float_abs)
    ret, gradients_bin = cv2.threshold(gradients, GRADIENT_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    return gradients_bin


def est_depth_diff(depth, bounds, est_depth, high = DEPTH_FINGER_THRESHOLD, low = DEPTH_NOISE_THRESHOLD):
    """
    Threshold depth image based on difference between true depth and
        estimated depth of RoI

    Args:
        depth (np.ndarray): True depth image
        bounds (np.ndarray): Bounds of RoI
        est_depth (function): Function to calculate depth vs. image coordinate

    Returns:
        TYPE: Description
    """
    bound_rect = cv2.boundingRect(bounds)
    x, y, w, h = bound_rect

    diff = np.zeros((h, w))

    # Calculate absolute depth difference between true depth image and
    #   estimated plane of RoI
    for i in range(w):
        for j in range(h):
            diff[j, i] = est_depth(x + i, y + j) - depth[y + j, x + i]

    # Threshold depth difference
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = (high < low) & (diff > low)

    mask[valid] = 255

    return mask

def est_depth_diff_mat(depth, est_depth, high = DEPTH_FINGER_THRESHOLD, low = DEPTH_NOISE_THRESHOLD):
    """
    Threshold depth image based on difference between true depth and
        estimated depth of RoI

    Args:
        depth (np.ndarray): True depth image
        bounds (np.ndarray): Bounds of RoI
        est_depth (function): Function to calculate depth vs. image coordinate

    Returns:
        TYPE: Description
    """
    h, w = depth.shape
    diff = np.zeros((h, w))

    # Calculate absolute depth difference between true depth image and
    #   estimated plane of RoI
    diff = est_depth - depth

    # Threshold depth difference
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = (diff < high) & (diff > low)

    mask[valid] = 255

    return mask



def planeFit(points):
    """
    Least squares fit of n-dimensional points to a plane

    Args:
        points (np.ndarray): N-dimensional array of data points;
            must be more points than dimensions

    Returns:
        (np.ndarray): Normal point to plane
        (np.ndarray): Plane equation
    """
    # Force overdetermined system
    assert(points.shape[0] > points.shape[1])
    ctr = points.T.mean(axis=1)
    x = points.T - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    return ctr, svd(M)[0][:, -1]


def draw_joints(img, offset, joint):
    """Summary

    Args:
        img (TYPE): Description
        offset (TYPE): Description
        joint (TYPE): Description

    Returns:
        TYPE: Description
    """
    x, y = offset
    j_x = int(joint[0] - x)
    j_y = int(joint[1] - y)

    if j_x < 0 or j_x >= img.shape[1]:
        return img
    if j_y < 0 or j_y >= img.shape[0]:
        return img

    return cv2.circle(img, (j_y, j_x), 2, (255, 0, 0), -1)

def filter_contours(depth, contours):
    best_contour = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_FINGER_AREA or area > MAX_FINGER_AREA:
            continue    

        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = min(float(w)/h, float(h)/w)
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        rect_area = w*h
        extent = float(area)/rect_area
        if extent < MIN_EXTENT:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / perimeter**2
        if circularity < MIN_CIRCULARITY or circularity > MAX_CIRCULARITY:
            continue

        if (area > best_area):
            best_contour = contour
    return best_contour

def find_finger_contour(depth, mask, contour):
    print(len(contour), contour.shape)
 
    height = depth.shape[0]
    width = depth.shape[1]

    # convexHull = cv2.convexHull(contour)
    hand = np.zeros((height,width), dtype = np.uint8)
    # center, radius = cv2.minEnclosingCircle(contour)
    # hand = cv2.drawContours(hand, contour, -1, (255, 255, 255), 3)
    cv2.fillPoly(hand, pts = contour, color=(255,255,255))

    kernel = np.ones((49,49),np.uint8)
    hand = cv2.dilate(hand, kernel, 1)

    gradients_float = cv2.Laplacian(depth, cv2.CV_64F, ksize=3)
    gradients_float_abs = gradients_float.clip(min=0)
    # gradients_float_abs = np.absolute(gradients_float)
    gradients = np.uint8(gradients_float_abs)
    gradients_bin = cv2.inRange(gradients, 0, 0.5)
    # cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ret, gradients_bin = cv2.threshold(gradients, 30, 255, cv2.THRESH_BINARY_INV)
    # ret, gradients_bin2 = cv2.threshold(gradients, 5, 255, cv2.THRESH_BINARY)

    fingers = cv2.bitwise_and(gradients_bin, hand)
    fingers = cv2.bitwise_and(fingers, mask)
    fingers = cv2.morphologyEx(fingers, cv2.MORPH_OPEN, kernel)



    cv2.namedWindow("hand")
    cv2.namedWindow("finger")
    cv2.namedWindow("finger mask")
    cv2.imshow('hand', cv2.resize(hand, (int(width/2.), int(height/2.))))
    cv2.imshow('finger', cv2.resize(fingers, (int(width/2.), int(height/2.))))
    cv2.imshow('finger mask', cv2.resize(mask, (int(width/2.), int(height/2.))))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test data sequence')

    # ====================================================
    #      Parameters for SortedTagDatabase instance
    # ====================================================

    # Required
    parser.add_argument(
        '--sequence_dir', metavar='STDB_SAVE_DIR',
        type=str, help='sub-directory of ./data/sequences \
            where RGBD/Joint data is saved', required=True)

    args = parser.parse_args()

    sequence_dir = os.path.join(SEQUENCE_DIRECTORY, args.sequence_dir)
    # sequence_dir = SEQUENCE_DIRECTORY + "/" + args.sequence_dir

    # RGBD data sequence
    rgb = load_data.load_rgbd(sequence_dir, data='rgb')
    d = load_data.load_rgbd(sequence_dir, data='depth')

    # Joint data sequence
    joints = load_data.load_csv(sequence_dir, data='joint')
    joints = np.array(joints)

    n_images = len(rgb)

    height = rgb[0].shape[0]
    width = rgb[0].shape[1]

    # Rescale joint data
    # joints = joints / JOINT_SCALE

    # # Flip image if joints are horizontally flipped (like RGB data)
    # if not JOINT_FLIP:
    #     joints[:, 2] = width - joints[:, 2]
    #     joints[:, 4] = width - joints[:, 4]
    #     joints[:, 6] = width - joints[:, 6]
    #     joints[:, 8] = width - joints[:, 8]

    # J1, J2, J3, J4 = [], [], [], []

    # # Extract data for each tracked joint
    # for i in range(n_images):
    #     j_1, j_2, j_3, j_4 = joints[i][2:4], joints[i][4:6], \
    #         joints[i][6:8], joints[i][8:10]
    #     J1.append(j_1)
    #     J2.append(j_2)
    #     J3.append(j_3)
    #     J4.append(j_4)

    # Get RoI bounds
    # bounds = get_bounds(rgb, N_CALIBRATION_FRAMES)

    roi = np.zeros((height,width), dtype = np.uint8)
    # cv2.fillPoly(roi, pts =[bounds], color=(255,255,255))
    roi = cv2.bitwise_not(roi)
    # bound_rect = cv2.boundingRect(bounds)
    # x, y, w, h = bound_rect

    # # Show RoI contour
    # img2 = cv2.drawContours(rgb[0], [bounds], -1, (0, 255, 0), 3)
    # cv2.imshow('ex_contour', img2)
    # cv2.imshow('ex_filled', roi)
    # cv2.waitKey(0)

    # T = get_transform_matrix(bounds)
    # est_plane = est_plane_equation(d[0], bounds)

    # est_plane_mat = np.zeros(d[0].shape)
    # for i in range(d[0].shape[0]):
    #     for j in range(d[0].shape[1]):
    #         est_plane_mat[i, j] = est_plane(j, i)

    # for i in range(10):
    #     plt_depth_diff_map(d[54+i], est_plane_mat)
    # plt_depth_diff(d[54], est_plane)

    est_plane_mat = d[0]

    for i in range(n_images):
        kernel = np.ones((5,5),np.uint8)

        depth = cv2.bitwise_and(d[i], d[i], mask = roi)
        # mask = est_depth_diff(d[i], bounds, est_plane)
        mask = est_depth_diff_mat(depth, est_plane_mat)
        gradient = depth_gradient(depth)
        gradient = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
        mask = cv2.bitwise_and(mask, gradient)

        
        # morph_gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        # opening = cv2.bitwise_and(mask, cv2.bitwise_not(morph_gradient))
        # opening = cv2.erode(mask, kernel, 1)
        opening = cv2.dilate(cv2.erode(mask, kernel, 2), kernel, 2)
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        opening = gaussian_filter(opening, sigma=1.5)

        modified_mask = opening

        __, contours, __ = cv2.findContours(
            modified_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = filter_contours(depth, contours)

        img = np.zeros((modified_mask.shape[0], modified_mask.shape[1], 3), dtype=np.uint8)
        if filtered_contours is not None:
            finger_mask = est_depth_diff_mat(depth, est_plane_mat, 35, 10)
            find_finger_contour(depth, finger_mask, filtered_contours)
            img = cv2.drawContours(img, filtered_contours, -1, (0, 0, 255), 3)
        

        # img = draw_joints(img, (x, y), J1[i])

        cv2.namedWindow("contours")   
        cv2.namedWindow("mask")   
        cv2.namedWindow("opening")   

        cv2.imshow('contours', cv2.resize(img, (int(width/2.), int(height/2.))))
        cv2.imshow('mask', cv2.resize(mask, (int(width/2.), int(height/2.))))
        cv2.imshow('opening', cv2.resize(opening, (int(width/2.), int(height/2.))))
        print(i)
        cv2.waitKey()

        # BEST_X = (x + RS.randint(w), y + RS.randint(h))
        # transform_coords(T, BEST_X)
