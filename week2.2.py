import numpy as np
import open3d as o3d

# --- 1. 准备数据 (同上) ---
H, W = 480, 640
rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
depth = np.random.randint(0, 5000, (H, W), dtype=np.uint16)
fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
depth_scale = 1000.0

# --- 2. 定义 ROI 区域 (左上角 u0,v0 到 右下角 u1,v1) ---
u0, v0, u1, v1 = 200, 150, 440, 330

# 截取 ROI
roi_depth = depth[v0:v1, u0:u1].astype(np.float64)
roi_rgb = rgb[v0:v1, u0:u1]

# --- 3. 生成网格坐标 ---
# uu 和 vv 是该区域内每个像素的 u, v 坐标
vv, uu = np.mgrid[v0:v1, u0:u1]

# --- 4. 反投影计算 XYZ ---
Z = roi_depth / depth_scale
X = (uu - cx) * Z / fx
Y = (vv - cy) * Z / fy

# --- 5. 过滤无效点 ---
# 条件：深度大于0，且不是无穷大，且距离在 10米以内
valid_mask = (roi_depth > 0) & np.isfinite(Z) & (Z < 10.0)

# 提取有效点的坐标和颜色
points = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=-1)
colors = roi_rgb[valid_mask] / 255.0  # Open3D 颜色需要归一化到 0-1

# --- 6. 创建 Open3D 点云对象并可视化 ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"生成了 {len(points)} 个点。正在打开可视化窗口...")
o3d.visualization.draw_geometries([pcd], window_name="ROI PointCloud")