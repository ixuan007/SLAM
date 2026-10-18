import numpy as np
import cv2
import open3d as o3d

# --- 1. 准备带有“脏数据”的模拟数据 ---
H, W = 480, 640
rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
depth = np.ones((H, W), dtype=np.uint16) * 2000  # 统一深度 2米

# 制造问题：
# 1. 无效深度 (0值)
depth[100:120, 100:120] = 0
# 2. 离群点 (极大的噪点)
depth[200, 200] = 50000

fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
depth_scale = 1000.0

# --- 2. 策略实施 ---

# 策略 A: 处理无效深度
# 将 NaN 或 Inf 设为 0
depth_clean = depth.astype(np.float64)
depth_clean[~np.isfinite(depth_clean)] = 0

# 策略 B: 处理前后景混合 (边缘抖动)
# 使用双边滤波，在平滑深度的同时保留边缘
# d=5: 邻域直径, sigmaColor=50: 颜色空间标准差, sigmaSpace=5: 坐标空间标准差
depth_clean = cv2.bilateralFilter(depth_clean.astype(np.float32), 5, 50, 5)

# --- 3. 生成点云 ---
vv, uu = np.mgrid[0:H, 0:W]
Z = depth_clean / depth_scale
# 过滤掉 <=0 和太远的点 (>10米)
valid = (depth_clean > 0) & (Z < 10.0)

X = (uu - cx) * Z / fx
Y = (vv - cy) * Z / fy
points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
colors = rgb[valid] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 策略 C: 处理离群点 (统计滤波)
# nb_neighbors=20: 查询周围20个点
# std_ratio=2.0: 距离超过标准差2倍的视为噪点移除
print("正在进行统计去噪...")
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# --- 4. 可视化对比 ---
# 原始点云(含噪点)设为红色，清洗后的设为原色
pcd.paint_uniform_color([1, 0, 0])

print("红色是原始含噪点云，彩色是清洗后的点云")
# o3d.visualization.draw_geometries([pcd, pcd_clean], window_name="Cleaning Strategy")
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Cleaning Strategy", width=800, height=600)
vis.add_geometry(pcd)
vis.add_geometry(pcd_clean)
vis.run()
vis.destroy_window()