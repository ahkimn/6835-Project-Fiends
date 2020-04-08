import argparse
import load_data
import os
import cv2
import numpy as np
from numpy.linalg import svd
# from matplotlib import pyplot as plt

from scipy.ndimage.filters import gaussian_filter

SEQUENCE_DIRECTORY = './data/sequences'
H_HIGH = 14
H_LOW = 3

MIN_PLANE = 10000
N_CALIBRATION_FRAMES = 10

DEPTH_NOISE_THRESHOLD = 30
FINGER_THRESHOLD = 55

RS = np.random.RandomState(seed=0)


def threshold_img(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0][:]
    hue = gaussian_filter(hue, sigma=7)

    mask = np.logical_and(hue < H_HIGH, hue >= H_LOW)
    hue[mask] = 255
    hue[~mask] = 0

    max_coverage = 0

    best_contour = None

    contours, _ = cv2.findContours(hue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)

        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        rect = cv2.boundingRect(approx)
        x, y, w, h = rect

        if w * h > MIN_PLANE:

            mask = np.zeros(hue.shape, np.uint8)
            mask[y:y + h, x:x + w] = hue[y:y + h, x:x + w]
            coverage = np.sum(mask)

            if coverage > max_coverage:
                best_contour = approx
                max_coverage = coverage

    return best_contour


def get_ordered_bounds(bounds):

    flat = bounds.reshape(4, 2)
    avg_x = np.mean(flat[:, 0])
    avg_y = np.mean(flat[:, 1])

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


# TODO could use desired aspect ratio (here we just using bounds of rectangle)
def get_transform_matrix(bounds, img, force_zero=False):

    bound_rect = cv2.boundingRect(bounds)
    x, y, w, h = bound_rect
    if force_zero:
        x, y = 0, 0

    p_1, p_2, p_3, p_4 = get_ordered_bounds(bounds)

    # upper_left
    X_1, Y_1 = x, y
    x_1, y_1 = p_1
    # upper right
    X_2, Y_2 = x + w, y
    x_2, y_2 = p_2
    # lower right
    X_3, Y_3 = x + w, y + h
    x_3, y_3 = p_3
    # lower left
    X_4, Y_4 = x, y + h
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


def est_plane_equation(depth, bounds, n_points=40):

    points = np.zeros((n_points, 3))
    height, width = depth.shape

    valid = 0

    while valid < n_points:

        x, y = RS.randint(width), RS.randint(height)

        if cv2.pointPolygonTest(bounds, (x, y), True) > 5:

            points[valid] = [x, y, depth[y, x]]
            valid += 1

    p, N = planeFit(points)

    def est_depth(x, y):

        z = (N[0] * (x - p[0]) + N[1] * (y - p[1]) - (p[2] * N[2]))
        z /= N[2]
        return -z

    return est_depth


def est_depth_diff(img, bounds, est_depth):

    bound_rect = cv2.boundingRect(bounds)
    x, y, w, h = bound_rect

    diff = np.zeros((h, w))

    for i in range(w):
        for j in range(h):
            diff[j, i] = est_depth(x + i, y + j) - img[y + j, x + i]

    mask = np.zeros((h, w), dtype=np.uint8)
    valid = (diff < FINGER_THRESHOLD) & (diff > DEPTH_NOISE_THRESHOLD)

    mask[valid] = 255


def planeFit(points):
    # Force overdetermined system
    assert(points.shape[0] > points.shape[1])
    ctr = points.T.mean(axis=1)
    x = points.T - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    return ctr, svd(M)[0][:, -1]


def get_bounds(rgb_list, n_frames):

    n_frames = min(len(rgb_list), n_frames)

    valid_contours = []

    for i in range(n_frames):
        c = threshold_img(rgb_list[i])
        if len(c) == 4:
            valid_contours.append(c)

    median_c = np.array([
        int(np.median([c.flatten()[i] for c in valid_contours]))
        for i in range(8)]).reshape(4, 1, 2)
    return median_c


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

    rgb = load_data.load_rgbd(sequence_dir, data='rgb')
    d = load_data.load_rgbd(sequence_dir, data='depth')

    # joints = load_data.load_csv(sequence_dir, data='joint')

    bounds = get_bounds(rgb, N_CALIBRATION_FRAMES)

    # img2 = cv2.drawContours(rgb[0], [bounds], -1, (0, 255, 0), 3)
    # cv2.imshow('u', img2)
    # cv2.waitKey(0)

    T = get_transform_matrix(bounds, rgb[0])
    est_plane = est_plane_equation(d[0], bounds)

    n_test = 400
    for i in range(n_test):

        est_depth_diff(d[i], bounds, est_plane)

    # img = cv2.drawContours(rgb[0], contours, -1, (0, 255, 0), 3)
    # cv2.imshow('a', img)
