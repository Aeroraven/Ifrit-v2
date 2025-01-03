# 0.002779 0.013149 -0.008682 22.269403
# -0.009061 -0.005879 -0.011804 -1.984316
# -0.012891 0.006967 0.006425 -40.171238
# 0.000000 0.000000 0.000000 1.000000

import numpy as np 

def normalize_by_scale(trs):
    scale = np.linalg.norm(trs[:3, :3], axis=1)
    scale = np.mean(scale)
    trs[:3, :3] /= scale
    return trs

def trs_to_euler(trs):
    # trs: 4x4 matrix
    # return: 3x1 euler angles
    r = np.arctan2(trs[2, 1], trs[2, 2])
    p = np.arctan2(-trs[2, 0], np.sqrt(trs[2, 1]**2 + trs[2, 2]**2))
    y = np.arctan2(trs[1, 0], trs[0, 0])
    return np.array([r, p, y])

def trs_to_euler2(trs):
    x = np.atan2(trs[2, 1], trs[2, 2])
    y = np.atan2(-trs[2, 0], np.sqrt(trs[2, 1]**2 + trs[2, 2]**2))
    z = np.atan2(trs[1, 0], trs[0, 0])
    return np.array([x, y, z])

def axisanglerot(axis, angle):
    # axis: 3x1 vector
    # angle: scalar
    # return: 3x3 rotation matrix
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([
        [c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y**2 * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c)]
    ])
    return rot

def euler_to_trs(euler):
    identity = np.eye(3)
    trs = identity
    Rx = axisanglerot(identity[0], euler[0])
    Ry = axisanglerot(identity[1], euler[1])
    Rz = axisanglerot(identity[2], euler[2])
    trs = np.matmul(Rx, trs)
    trs = np.matmul(Ry, trs)
    trs = np.matmul(Rz, trs)
    return trs


if __name__ == '__main__':
    trs = np.array([
        [0.002779, 0.013149, -0.008682],
        [-0.009061, -0.005879, -0.011804],
        [-0.012891, 0.006967, 0.006425],
    ])
    trs_norm = normalize_by_scale(trs)
    print(trs_norm)
    euler = trs_to_euler2(trs_norm)
    print(euler_to_trs(euler))
    
    # [ 0.013149 -0.009061 -0.012891]