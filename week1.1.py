import numpy as np
#作业1

# 1. 定义一个函数：从欧拉角(roll, pitch, yaw)生成旋转矩阵
def euler_to_rotation(roll, pitch, yaw):
    # 这里假设是 ZYX 顺序的内旋
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    return R


# 2. 定义一个函数：从旋转矩阵提取四元数 (w, x, y, z)
def rotation_to_quaternion(R):
    tr = np.trace(R)
    q = np.zeros(4)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    # ... (省略其他分支，完整公式需处理所有情况)

    return np.array([qw, qx, qy, qz])


# 3. 构建 SE3 矩阵
def create_se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# 测试一下
R_test = euler_to_rotation(0.1, 0.2, 0.3)
t_test = np.array([1.0, 2.0, 3.0])
T_test = create_se3(R_test, t_test)
print("生成的 SE3 矩阵:\n", T_test)

q = rotation_to_quaternion(R_test)
print("四元数 (w, x, y, z):", q)