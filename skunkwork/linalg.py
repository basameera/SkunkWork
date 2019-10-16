"""
SkunkWork Linear Algebra
========================


get matrix stuff from 
/home/sameera/Github/msc-thesis-work/Feature tracking test/plot_3D.py'

"""
import numpy as np

__all__ = ["isRotationMatrix", "eulerAnglesToRotationMatrix",
           "rotationMatrixToEulerAngles"]


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def eulerAnglesToRotationMatrix(theta, deg=False):
    """euler convenstion = X-Y-Y (alpha, beta, gamma)

    Arguments:
        theta {int array} -- [(alpha, beta, gamma)]

    Returns:
        Rotation matrix (3x3)
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


def rotationMatrixToEulerAngles(R):
    """
    X-Y-Z
    =====

    Arguments:
        R {3x3} -- [3x3]

    Returns:
        x, y, z -- Euler angles
    """
    assert(isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


if __name__ == "__main__":
    pass
