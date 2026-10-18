import numpy as np
import open3d as o3d

# --- 1. 准备数据 ---
H, W = 480, 640
rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
depth = np.random.randint(1000, 3000, (H, W), dtype=np.uint16)  # 给点有效深度
fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
depth_scale = 1000.0


def generate_pcd(rgb, depth, fx, fy, cx, cy, scale):
    """辅助函数：根据给定的参数生成点云"""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    Z = depth.astype(np.float64) / scale
    valid = (depth > 0) & (Z < 10.0)

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
    colors = rgb[valid] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# --- 2. 生成三种失真的点云 ---

# 情况 A: fx 变小 (视野变宽，物体看起来被拉宽了)
pcd_wide = generate_pcd(rgb, depth, fx * 0.5, fy, cx, cy, depth_scale)
pcd_wide.paint_uniform_color([1, 0, 0])  # 红色

# 情况 B: depth_scale 变小 (Z 变大，物体看起来变远了)
pcd_far = generate_pcd(rgb, depth, fx, fy, cx, cy, depth_scale * 0.5)
pcd_far.paint_uniform_color([0, 1, 0])  # 绿色

# 情况 C: cx 偏移 (点云整体横向偏移)
pcd_shifted = generate_pcd(rgb, depth, fx, fy, cx + 100, cy, depth_scale)
pcd_shifted.paint_uniform_color([0, 0, 1])  # 蓝色

# --- 3. 同时显示进行对比 ---
print("红色: fx变小(变宽), 绿色: scale变小(变远), 蓝色: cx偏移")
o3d.visualization.draw_geometries([pcd_wide, pcd_far, pcd_shifted], window_name="Distortion Comparison")