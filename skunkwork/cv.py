import cv2 as cv
import numpy as np


def image_resize(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def crop_image_from_center(img, c, size=200):
    x = c[0]-(size//2)
    y = c[1]-(size//2)

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    w = h = size
    return img[y:y+h, x:x+w]


def kernal_blur(size=7):
    return np.ones((size, size), dtype="float") * (1.0 / (size * size))


def kernal_sharpen():
     # construct a sharpening filter
    return np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")


def kernal_laplacian():
    return np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")


def kernal_edge():
    return np.array((
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]), dtype="int")


def kernal_sobel_x():
    return kernal_sobel_x_right()


''' backup
def kernal_sobel_x_right():
    return np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")'''


def kernal_sobel_x_right():
    return np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")


def kernal_sobel_x_left():
    return np.array((
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]), dtype="int")


def kernal_sobel_y():
    return kernal_sobel_y_bottom()


def kernal_sobel_y_bottom():
    """bottom edge

    Returns:
        [type] -- [description]
    """
    return np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")


def kernal_sobel_y_top():
    """top edge

    Returns:
        [type] -- [description]
    """
    return np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]), dtype="int")


def normalize_kernal(kernal):
    return kernal/np.sum(np.abs(kernal))


def kernal_custom():
    """top edge

    Returns:
        [type] -- [description]
    """
    return np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]), dtype="int")


def kernal_sobel_xy():
    return kernal_sobel_x() + kernal_sobel_y()


def kernal_sobel_bottom_right():
    return kernal_sobel_y_bottom() + kernal_sobel_x_right()


def kernal_sobel_bottom_left():
    return kernal_sobel_y_bottom() + kernal_sobel_x_left()


def convolution(gray, kernal):

    if isinstance(kernal, list):
        for k in kernal:
            gray = cv.filter2D(gray, -1, k)
        return gray
    elif isinstance(kernal, np.ndarray):
        return cv.filter2D(gray, -1, kernal)


def hsv_filter(frameBGR, out_morph=True, morph_kernal_size=5):

    # icol = (105, 30, 0, 177, 255, 180)  # old

    # icol = (105, 34, 0, 255, 255, 255)  # sim final
    icol = (105, 15, 15, 177, 255, 255)  # real final

    # frameBGR = cv.GaussianBlur(frameBGR, (7, 7), 0)
    # frameBGR = cv.medianBlur(frameBGR, 7)
    frameBGR = cv.bilateralFilter(frameBGR, 25, 75, 75)
    """kernal = np.ones((15, 15), np.float32)/255
    frameBGR = cv.filter2D(frameBGR, -1, kernal)"""

    # HSV (Hue, Saturation, Value).
    # Convert the frame to HSV colour model.
    hsv = cv.cvtColor(frameBGR, cv.COLOR_BGR2HSV)

    # HSV values to define a colour range.
    colorLow = np.array([icol[0], icol[1], icol[2]])
    colorHigh = np.array([icol[3], icol[4], icol[5]])
    mask = cv.inRange(hsv, colorLow, colorHigh)

    if out_morph:
        # kernal_size = 21
        kernal = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (morph_kernal_size, morph_kernal_size))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernal)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal)

    # Show morphological transformation mask
    # cv.imshow('mask', mask)
    return [mask, frameBGR]


def hstack(imgs):
    out = imgs[0]
    for n in range(1, len(imgs)):
        out = np.hstack((out, imgs[n]))
    return out


def eulerAnglesToRotationMatrix(theta, deg=False):
    """## euler convenstion = X-Y-Y (alpha, beta, gamma)

    Arguments:
        theta {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    if deg:
        theta = np.radians(theta)

    R_x = np.array([[1,         0,                  0],
                    [0,         np.cos(theta[0]), -np.sin(theta[0])],
                    [0,         np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])],
                    [0,                     1,      0],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def warp_imgs(img_n, euler=[0, 0, 0], show=False):
    '''
    euler: x, y, z
    '''
    (h, w) = img_n.shape

    euler = np.array(euler, dtype=np.float)
    # 1
    R = eulerAnglesToRotationMatrix(euler, deg=True)

    translation = np.array([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0, 1]
    ], dtype=np.float)

    trans = np.matmul(R, translation)

    # 4 - z offset
    trans[2, 2] += h

    camera_matrix = np.array([
        [h, 0, h/2],
        [0, h, h/2],
        [0, 0, 1]
    ], dtype=np.float)

    # 5
    transform = np.matmul(camera_matrix, trans)

    # 6
    warp_img = cv.warpPerspective(img_n, transform, dsize=(
        w, h), flags=cv.INTER_LINEAR, borderValue=(0., 0.))

    if show:
        img_stack = np.hstack((img_n, warp_img))
        cv.imshow('img, warp', img_stack)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return warp_img


if __name__ == "__main__":
    pass
