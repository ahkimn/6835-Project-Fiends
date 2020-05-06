import cv2
import numpy as np

mouseX = -1
mouseY = -1
mouse_down = False

def get_depth(event, x, y, flags,param):
    global mouseX, mouseY
    global mouse_down

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False

    if mouse_down:
        print(depth[y, x])
        mouseX, mouseY = x, y


# Main demo code
if __name__ == "__main__":
    # path = "Z:/Dropbox/Dropbox (MIT)/MIT/2020/Spring/6.835/Project/Code/rgbd_scan/rgbd_scan/data/box2/depth/depth25.png"
    path = "Z:/Dropbox/Dropbox (MIT)/MIT/2020/Spring/6.835/Project/Code/rgbd_scan/rgbd_scan/rgbd_scan/test0/depth/depth40.png"
    depth = cv2.imread(path, -1)

    min_val, max_val, __, __ = cv2.minMaxLoc(depth)
    print(min_val, max_val)
    min_val = 0     #Fixed here to actual show range
    depth_scaled = cv2.convertScaleAbs(depth, alpha=255/(max_val-min_val))
    depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("depth", get_depth)


    while(1):
        cv2.imshow("depth", depth_colored)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            cv2.circle(depth_colored, (mouseX, mouseY),1,(0,0,255),-1)
    # cv2.waitKey()
    cv2.destroyAllWindows()