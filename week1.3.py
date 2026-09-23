"""
camera -> base -> map 三段变换链
将 20 个三维点从 camera 转到 map，再逆变换回来
"""
"""
我的理解
坐标系链  A -> B -> C
点表示为   pA -> pB -> pC

其中：
  pA · TA = pB     （A 点用 TA 变换到 B）
  pB · TB = pC     （B 点用 TB 变换到 C）

则：
  pA · TA · TB = pC

再逆变换回来：
  pA = pC · (TA · TB)⁻¹

所以验证 pA = pA 就行


"""
import numpy as np

#就这个问题，我直接生成两个合法的T矩阵，用于验证此问题
np.random.seed(0)#每次的随机都一样
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
R1 = xuanzhuan_R()
R2 = xuanzhuan_R()
t1 = np.random.randn(3)
t2 = np.random.randn(3)
Tcb = random_T(R1,t1)
Tbm = random_T(R2,t2)
#生成20个点，转为齐次坐标 (4x20)
p_c = np.random.randn(3, 20)
p_c_homo = np.vstack([p_c, np.ones(20)])
# np.ones((1, 20))：生成形状 (1, 20) 的一行全 1
# np.vstack([A, B])：上下叠起来，把 (3,20) 和 (1,20) 叠成 (4,20)
Tcm = Tbm @ Tcb     #@表示左乘
p_m_h = Tcm @ p_c_homo
p_m = p_m_h[:3,:]

Tmc = np.linalg.inv(Tcm)
p_c_b_h = Tmc @ p_m_h
p_c_b = p_c_b_h[:3,:]
err_each = np.linalg.norm(p_c - p_c_b, axis=0)   # (20,)

print("每个点的误差：")
for i, e in enumerate(err_each):
    print(f"  第{i+1:2d}个点：{e:.3e}")
print("是否全部通过：", err_each.max() < 1e-10)
if np.all(err_each < 1e-10):
    print("变换链验证成功！")




