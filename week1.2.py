import numpy as np

np.random.seed(0)
#每次代码生成的随机旋转和平移序列都一样，方便复现
def xuanzhuan_R():
    A = np.random.randn(3,3)
    Q,R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def random_T(R,t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

max_err = 0.0
all_ok = True

for i in range(100):
    R = xuanzhuan_R()
    t = np.random.randn(3)
    T = random_T(R,t)
    T_inv = np.linalg.inv(T)
    I = T @ T_inv
    err = np.max(np.abs(I - np.eye(4)))
    max_err = max(max_err,err)
    print(f"f[FAIL]第{i + 1}组误差：{err:.3e}")

    if err > 1e-10:
        all_ok = False
        print(f"f[FAIL]第{i+1}组误差：{err:.3e}")
print(f"完成 100 组测试")
print(f"最大误差: {max_err:.3e}")
print(f"是否全部满足 T·T⁻¹≈I: {all_ok}")