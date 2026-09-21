import numpy as np

def oula(roll, pitch, yaw):
# roll绕x轴  pitch绕y轴  yaw绕z轴
    cr,sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])
    return R

#提取四元数
def four(R):
    t0 = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
    t1 = 1.0 + R[0, 0] - R[1, 1] - R[2, 2]
    t2 = 1.0 - R[0, 0] + R[1, 1] - R[2, 2]
    t3 = 1.0 - R[0, 0] - R[1, 1] + R[2, 2]

    # 找到最大的那个
    t = [t0, t1, t2, t3]
    idx = np.argmax(t)

    if idx == 0:  # t0 最大，先算 w
        w = 0.5 * np.sqrt(t0)
        x = (R[2, 1] - R[1, 2]) / (4.0 * w)
        y = (R[0, 2] - R[2, 0]) / (4.0 * w)
        z = (R[1, 0] - R[0, 1]) / (4.0 * w)

    elif idx == 1:  # t1 最大，先算 x
        x = 0.5 * np.sqrt(t1)
        w = (R[2, 1] - R[1, 2]) / (4.0 * x)
        y = (R[0, 1] + R[1, 0]) / (4.0 * x)
        z = (R[0, 2] + R[2, 0]) / (4.0 * x)

    elif idx == 2:  # t2 最大，先算 y
        y = 0.5 * np.sqrt(t2)
        w = (R[0, 2] - R[2, 0]) / (4.0 * y)
        x = (R[0, 1] + R[1, 0]) / (4.0 * y)
        z = (R[1, 2] + R[2, 1]) / (4.0 * y)

    else:  # t3 最大，先算 z
        z = 0.5 * np.sqrt(t3)
        w = (R[1, 0] - R[0, 1]) / (4.0 * z)
        x = (R[0, 2] + R[2, 0]) / (4.0 * z)
        y = (R[1, 2] + R[2, 1]) / (4.0 * z)

    return np.array([w,x,y,z])


def SE3(R,t):
    T = np.eye(4)#生成单位矩阵
    T[:3, :3] = R
    T[:3, 3] = t
    return T

R_test = oula(0.1,0.2,0.3)
t_test = np.array([1.0, 2.0, 3.0])
T_test = SE3(R_test, t_test)
F_test = four(R_test)
print("四元数为\n",F_test)
print("SE3矩阵为\n",T_test)
print(np.round(R_test,8))
#四元数到旋转矩阵
def four_R(w1,x1,y1,z1):
    R = np.array([
        [1 - 2*(y1*y1 + z1*z1),  2*(x1*y1 - w1*z1),      2*(x1*z1 + w1*y1)],
        [2*(x1*y1 + w1*z1),      1 - 2*(x1*x1 + z1*z1),  2*(y1*z1 - w1*x1)],
        [2*(x1*z1 - w1*y1),      2*(y1*z1 + w1*x1),      1 - 2*(x1*x1 + y1*y1)]
    ])
    return R

R_test1 = four_R(np.cos(np.pi/4), 0, 0, np.sin(np.pi/4))
print("得到的矩阵为：\n", np.round(R_test1, 8))

def SE3_R(T1):
    R1 = T1[:3,:3]
    return R1
R1_test = SE3_R(T_test)
print("得到的矩阵为：\n", np.round(R1_test, 8))
