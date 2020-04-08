"""
SkunkWork 3D plotting
=====================


Using matplotlib

/home/sameera/Github/msc-thesis-work/Feature tracking test/plot_3D.py

"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from .linalg import *

__all__ = ["draw_3D_ref_frames_headless", "get_origin_point", "plot_3d_drone", "sample_3d_ref_plotting",
           "get_world_frame_cont_data", "get_point_array_world", "get_drone_points", "get_drone_array_world"]


def get_origin_point(scale=1.0):
    return np.vstack(
        (np.zeros(shape=(1, 3)),
         np.identity(3)*scale)
    )


def getAxes(unit_ax=1.0):

    a_x, a_y, a_z = np.eye(3)*unit_ax

    a_x = a_x.reshape(1, -1)
    a_y = a_y.reshape(1, -1)
    a_z = a_z.reshape(1, -1)

    return a_x, a_y, a_z


def rotate_axes(point, R, unit_ax=1.0):

    # rotate axes
    a_x, a_y, a_z = getAxes(unit_ax=unit_ax)

    a_x = np.matmul(R, a_x.T).T + point
    a_y = np.matmul(R, a_y.T).T + point
    a_z = np.matmul(R, a_z.T).T + point

    ux = np.concatenate((point, a_x), axis=0).T
    uy = np.concatenate((point, a_y), axis=0).T
    uz = np.concatenate((point, a_z), axis=0).T
    return ux, uy, uz


def get_world_frame_cont_data(factor_T=1.0):
    size = 4
    t = np.empty(shape=(size, 1, 3))
    R = np.empty(shape=(size, 3, 3))

    t[0] = np.zeros(shape=(1, 3))
    R[0] = np.identity(3)

    # change w/ z
    # x, y, z
    alpha, beta, gamma = 0, 0, 30
    euler_angles = np.radians([alpha, beta, gamma])
    rot = eulerAnglesToRotationMatrix(euler_angles)
    t[1] = np.array([1, 0, 0]).reshape(1, -1)*factor_T
    # t[1] = np.ones((1, 3))*0.5
    if isRotationMatrix(rot):
        R[1] = rot

    # change w/ y
    # x, y, z
    alpha, beta, gamma = 0, 0, 60
    euler_angles = np.radians([alpha, beta, gamma])
    rot = eulerAnglesToRotationMatrix(euler_angles)
    t[2] = np.array([2, 0, 0]).reshape(1, -1)*factor_T
    # t[2] = np.ones((1, 3))*1.0
    if isRotationMatrix(rot):
        R[2] = rot

    # change w/ x
    # x, y, z
    alpha, beta, gamma = 0, 0, 90
    euler_angles = np.radians([alpha, beta, gamma])
    rot = eulerAnglesToRotationMatrix(euler_angles)
    t[3] = np.array([3, 0, 0]).reshape(1, -1)*factor_T
    # t[3] = np.ones((1, 3))*1.5
    if isRotationMatrix(rot):
        R[3] = rot

    return R, t


def draw_3D_ref_frames_with_Rt(translations, rotations, unit_ax=1.0, scale_axes=True, show_label=True, figsize=None, show_plot=False):
    """Previously: draw_3D_ref_frames

    Arguments:
        translations {[type]} -- [description]
        rotations {[type]} -- [description]

    Keyword Arguments:
        unit_ax {float} -- [description] (default: {1.0})
        scale_axes {bool} -- [description] (default: {True})
        show_label {bool} -- [description] (default: {True})
        figsize {[type]} -- [description] (default: {None})
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for n in range(translations.shape[0]):

        point = translations[n]

        assert(point.shape == (1, 3))

        ux, uy, uz = rotate_axes(point, rotations[n], unit_ax=unit_ax)

        ax.plot(ux[0], ux[1], ux[2], marker='.', c='r')
        ax.plot(uy[0], uy[1], uy[2], marker='.', c='g')
        ax.plot(uz[0], uz[1], uz[2], marker='.', c='b')
        if show_label:
            ax.text(point[0, 0], point[0, 1], point[0, 2], str(n))

    if scale_axes:
        # get axis limits
        max_scale = np.max(np.vstack((np.diff(np.array(ax.get_xlim())), np.diff(
            np.array(ax.get_ylim())), np.diff(np.array(ax.get_zlim3d())))))

        # set axis limits
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[0] + max_scale)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0] + max_scale)
        ax.set_zlim(ax.get_zlim3d()[0], ax.get_zlim3d()[0] + max_scale)

    # show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if show_plot:
        plt.show()


def draw_3D_ref_frames_headless(point_array, drone_array=None, drone_shape='x', scale_axes=True, show_label=True, figsize=None, show_plot=True, title='',
                                marker='.', start_marker_size=1, drone_scale=1.0, tight=False, save_filename=None):
    """[summary]

    Arguments:
        point_array {[type]} -- [n, 4, 3]

    Keyword Arguments:
        drone_array {[type]} -- [description] (default: {None})
        drone_shape {str} -- [description] (default: {'x'})
        scale_axes {bool} -- [description] (default: {True})
        show_label {bool} -- [description] (default: {True})
        figsize {[type]} -- [description] (default: {None})
        show_plot {bool} -- [description] (default: {True})
        title {str} -- [description] (default: {''})
        marker {str} -- [description] (default: {'.'})
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    point = point_array[0]
    ax.scatter(point[0, 0], point[0, 1], point[0, 2], marker='*',
               c='magenta', linewidths=start_marker_size)

    for n in range(point_array.shape[0]):
        assert(point_array[n].shape == (4, 3))

        point, a_x, a_y, a_z = point_array[n].reshape(4, 1, 3)

        ux = np.concatenate((point, a_x), axis=0).T
        uy = np.concatenate((point, a_y), axis=0).T
        uz = np.concatenate((point, a_z), axis=0).T

        ax.plot(ux[0], ux[1], ux[2], marker=marker, c='r')
        ax.plot(uy[0], uy[1], uy[2], marker=marker, c='g')
        ax.plot(uz[0], uz[1], uz[2], marker=marker, c='b')
        if show_label:
            ax.text(point[0, 0], point[0, 1], point[0, 2], str(n))

    if drone_array is not None:
        for n in range(drone_array.shape[1]):
            dx = drone_array[0, n]
            dy = drone_array[1, n]
            draw_drone(ax, dx, dy, drone_shape=drone_shape,
                       drone_scale=drone_scale)
    if tight:
        plt.tight_layout()
    if scale_axes:

        # get axis limits
        max_scale = np.max(np.vstack((np.diff(np.array(ax.get_xlim())), np.diff(
            np.array(ax.get_ylim())), np.diff(np.array(ax.get_zlim3d())))))

        # set axis limits
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[0] + max_scale)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0] + max_scale)
        ax.set_zlim(ax.get_zlim3d()[0], ax.get_zlim3d()[0] + max_scale)

    # show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.legend(['X', 'Y', 'Z'])

    if save_filename is not None:
        plt.savefig(save_filename, dpi=300)

    if show_plot:
        plt.show()


def test_3d_plotting():
    transforms = np.random.random(size=(5, 1, 3))*20
    transforms = transforms.astype(np.int)
    rotations = np.empty(shape=(transforms.shape[0], 3, 3))

    for n in range(rotations.shape[0]):
        rotations[n] = np.identity(3) * np.power(-1, n)

    draw_3D_ref_frames_with_Rt(transforms, rotations)


def get_drone_points(drone_scale=1.0, drone_shape='x'):
    """[summary]

    Keyword Arguments:
        drone_scale {float} -- [description] (default: {1.0})
        drone_shape {str} -- [description] (default: {'x'})

    Returns:
        dx -- [3, 2]
        dy -- [3, 2]

    """
    dx, dy, _ = np.identity(3)*drone_scale

    dx = np.array([dx, -dx]).T
    dy = np.array([dy, -dy]).T

    if drone_shape == 'x':
        alpha, beta, gamma = 0, 0, 45
        euler_angles = np.radians([alpha, beta, gamma])
        rot = eulerAnglesToRotationMatrix(euler_angles)
        dx = np.matmul(rot, dx)
        dy = np.matmul(rot, dy)

    return dx, dy


def draw_drone(ax, dx, dy, color='gray', linewidth=2, marker='o', markersize=2, drone_shape='x', drone_scale=1.0):
    """[summary]

    Arguments:
        ax {[type]} -- [plot ax]
        dx {[type]} -- [3, 2]
        dy {[type]} -- [3, 2]

    Keyword Arguments:
        color {str} -- [description] (default: {'indigo'})
        linewidth {int} -- [description] (default: {2})
        marker {str} -- [description] (default: {'o'})
        markersize {int} -- [description] (default: {20})
        drone_shape {str} -- [description] (default: {'x'})
    """
    # linewidth *= drone_scale
    markersize *= drone_scale
    if markersize > 5.0:
        markersize = 5

    ax.plot(dx[0], dx[1], dx[2], marker='',
            c=color, linewidth=linewidth, markersize=markersize)
    ax.plot(dy[0], dy[1], dy[2], marker='',
            c=color, linewidth=linewidth, markersize=markersize)
    ax.scatter(dx[0, 0], dx[1, 0], dx[2, 0], marker=marker, c='orange',
               linewidths=markersize)
    ax.scatter(dx[0, 1], dx[1, 1], dx[2, 1], marker=marker, c='black',
               linewidths=markersize)

    ax.scatter(dy[0, 0], dy[1, 0], dy[2, 0], marker=marker, c='black',
               linewidths=markersize)

    if drone_shape == '+':
        color = 'black'
    elif drone_shape == 'x':
        color = 'orange'

    ax.scatter(dy[0, 1], dy[1, 1], dy[2, 1], marker=marker, c=color,
               linewidths=markersize)


def Rt_drone(dx, dy, R, t):
    """[summary]

    Arguments:
    ---
        dx {[type]} -- [3,2]
        dy {[type]} -- [3,2]
        R {[type]} -- [3,3]
        t {[type]} -- [1,3]

    Returns:
    ---
        [type] -- [description]
    """
    assert(R.shape == (3, 3))
    assert(t.shape == (1, 3))
    dx = np.matmul(R, dx)
    dy = np.matmul(R, dy)
    dx[:, 0] = dx[:, 0] + t.T[:, 0]
    dx[:, 1] = dx[:, 1] + t.T[:, 0]
    dy[:, 0] = dy[:, 0] + t.T[:, 0]
    dy[:, 1] = dy[:, 1] + t.T[:, 0]
    return dx, dy


def plot_3d_drone():

    scale_axes = True
    show_label = False
    figsize = (10, 8)
    title = 'drone'
    show_plot = 1
    marker = ''
    start_end_marker_width = 10

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # ref-axis
    scale = 0.3
    init_ref_frame_point = np.vstack(
        (np.zeros(shape=(1, 3)), np.identity(3)*scale))

    point, a_x, a_y, a_z = init_ref_frame_point.reshape(4, 1, 3)

    ux = np.concatenate((point, a_x), axis=0).T
    uy = np.concatenate((point, a_y), axis=0).T
    uz = np.concatenate((point, a_z), axis=0).T

    # start-end point markers
    ax.scatter(point[0, 0], point[0, 1], point[0, 2], marker='*',
               c='orange', linewidths=start_end_marker_width)
    ax.scatter(1, 1, 1, marker='*', c='black',
               linewidths=start_end_marker_width)

    ax.plot(ux[0], ux[1], ux[2], marker=marker, c='r', label='x')
    ax.plot(uy[0], uy[1], uy[2], marker=marker, c='g', label='y')
    ax.plot(uz[0], uz[1], uz[2], marker=marker, c='b', label='z')

    ax.legend()

    # drone
    drone_scale = .5
    drone_shape = 'x'
    dx, dy = get_drone_points(drone_scale, drone_shape=drone_shape)

    R, t = _get_world_frame_cont_data()
    # drone_array = get_drone_array_world(dx, dy, R, t)
    # print('drone_array', drone_array.shape)

    # rotate and translate the drone
    # dx = np.matmul(R, dx) + t
    # print('dx:', dx.shape)

    print(R[0].shape, t[0].shape)

    n = 2

    dx, dy = Rt_drone(dx, dy, R[n], t[n])

    draw_drone(ax, dx, dy, drone_shape=drone_shape)

    if show_label:
        ax.text(point[0, 0], point[0, 1], point[0, 2], str(n))

    if scale_axes:
        # get axis limits
        max_scale = np.max(np.vstack((np.diff(np.array(ax.get_xlim())), np.diff(
            np.array(ax.get_ylim())), np.diff(np.array(ax.get_zlim3d())))))

        # set axis limits
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[0] + max_scale)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[0] + max_scale)
        ax.set_zlim(ax.get_zlim3d()[0], ax.get_zlim3d()[0] + max_scale)

    # show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    if show_plot:
        plt.show()
    # print('sdfsfsddfsf')


def _get_world_frame_cont_data(factor_T=1.0):
    size = 4
    t = np.empty(shape=(size, 1, 3))
    R = np.empty(shape=(size, 3, 3))

    t[0] = np.zeros(shape=(1, 3))
    R[0] = np.identity(3)

    # change w/ z
    # x, y, z
    alpha, beta, gamma = 0, 0, 30
    euler_angles = np.radians([alpha, beta, gamma])
    rot = eulerAnglesToRotationMatrix(euler_angles)
    t[1] = np.array([1, 0, 0]).reshape(1, -1)*factor_T
    # t[1] = np.ones((1, 3))*0.5
    if isRotationMatrix(rot):
        R[1] = rot

    # change w/ y
    # x, y, z
    alpha, beta, gamma = 0, 0, 60
    euler_angles = np.radians([alpha, beta, gamma])
    rot = eulerAnglesToRotationMatrix(euler_angles)
    t[2] = np.array([2, 0, 0]).reshape(1, -1)*factor_T
    # t[2] = np.ones((1, 3))*1.0
    if isRotationMatrix(rot):
        R[2] = rot

    # change w/ x
    # x, y, z
    alpha, beta, gamma = 0, 0, 90
    euler_angles = np.radians([alpha, beta, gamma])
    rot = eulerAnglesToRotationMatrix(euler_angles)
    t[3] = np.array([3, 0, 0]).reshape(1, -1)*factor_T
    # t[3] = np.ones((1, 3))*1.5
    if isRotationMatrix(rot):
        R[3] = rot

    return R, t


def get_point_array_world(init_ref_frame_point, R_w, t_w):
    point_array = np.empty(shape=(R_w.shape[0], 4, 3))

    for n in range(R_w.shape[0]):

        point_array[n] = np.matmul(R_w[n], init_ref_frame_point.T).T + t_w[n]

    return point_array


def get_drone_array_world(dx, dy, R_w, t_w):
    drone_array = np.empty(shape=(2, R_w.shape[0], 3, 2))

    for n in range(R_w.shape[0]):
        drone_array[0, n], drone_array[1, n] = Rt_drone(dx, dy, R_w[n], t_w[n])
    return drone_array


def sample_3d_ref_plotting(size = 10):
    t = np.empty(shape=(size, 1, 3))
    R = np.empty(shape=(size, 3, 3))

    t[0] = np.ones(shape=(1, 3))*0.5
    R[0] = np.identity(3)

    scale = 1.0
    init_ref_frame_point = np.vstack(
        (np.zeros(shape=(1, 3)), np.identity(3)*scale))

    n_points = 2
    point_array = np.empty(shape=(n_points, 4, 3))

    point_array[0] = init_ref_frame_point
    point_array[1] = np.matmul(
        R[0], init_ref_frame_point.T).T + t[0]

    draw_3D_ref_frames_headless(
        point_array, title="world")


if __name__ == "__main__":
    # test_3d_plotting()
    # plot_3d_drone()

    # get test R, t
    R, t = _get_world_frame_cont_data(factor_T=100)

    # init ref point
    # scale = 1.0
    scale = 0.5
    init_ref_frame_point = np.vstack(
        (np.zeros(shape=(1, 3)), np.identity(3)*scale))

    # get point array
    point_array_world = get_point_array_world(init_ref_frame_point, R, t)

    # drone
    drone_scale = scale*1.33
    drone_shape = 'x'
    dx, dy = get_drone_points(drone_scale, drone_shape=drone_shape)

    drone_array = get_drone_array_world(dx, dy, R, t)

    # drawing
    draw_3D_ref_frames_headless(
        point_array_world, drone_array=drone_array, drone_shape=drone_shape, show_plot=True, title="world", start_marker_size=1, drone_scale=drone_scale)
