from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import pyautogui

DATA_DIR = 'data/points.txt'
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SCALE = 100
MIN_POINT_DIST = 30
ERROR_LIMIT = 50

def scale_point(point):
    return [point[0] / SCALE * SCREEN_WIDTH, point[1] / SCALE * SCREEN_HEIGHT]

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse(points, d=MIN_POINT_DIST):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if points[i][0] < 0 or points[i][1] < 0:
            ret.append((points[i][0], points[i][1]))
            taken[i] = True
            continue
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if points[j][0] < 0 or points[j][1] < 0:
                    continue
                elif dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret


def lin_fit_points(points, window=5):
    lin_reg = LinearRegression()

    fitted_points = []
    for i in range(len(points) - window):
        data_slice = points[i: i+window]

        Xy = np.array(data_slice)
        X, y = np.split(Xy,[-1],axis=1)
        lin_reg.fit(X, y)

        y_hat = lin_reg.predict(X)

        mse = metrics.mean_squared_error(y, y_hat)

        if mse < ERROR_LIMIT:
            new_points = [[i[0], j[0]] for i, j in zip(X, y_hat)]
            fitted_points.extend(new_points)
        else:
            fitted_points.extend(points[i:i+window])

    return fitted_points


def read_points(dir=DATA_DIR):
    f = open(dir)
    points = [[float(i.strip()) for i in s.split(',')] for s in f.readlines()]
    scaled_points = [scale_point(i) for i in points]

    filtered_points = fuse(scaled_points)
    return filtered_points

def main():
    points = read_points()

    reposition_cursor = False
    frame = 0
    for x, y in points:
        if x < 0 or y < 0:
            reposition_cursor = True
        elif y > SCREEN_HEIGHT * 0.9:
            pyautogui.click(x, y)
            reposition_cursor = True
        else:
            if reposition_cursor:
                pyautogui.moveTo(x, y)
                reposition_cursor = False
            pyautogui.dragTo(x, y, 0.2, button='left')

        im2 = pyautogui.screenshot(f'data/ui_demo/frame_{frame}.png')
        frame += 1


if __name__ == '__main__':
    main()