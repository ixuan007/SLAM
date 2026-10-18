import numpy as np
import cv2

# --- 1. 模拟加载数据 (实际使用时请替换为你的真实图片路径) ---
# 假设图片大小为 640x480
H, W = 480, 640
rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
# 模拟深度图，数值代表毫米
depth = np.random.randint(0, 5000, (H, W), dtype=np.uint16)

# 相机内参 (模拟值，实际请从 CameraInfo 读取)
fx, fy = 500.0, 500.0
cx, cy = 320.0, 240.0
depth_scale = 1000.0  # 1000 表示深度图单位是毫米


# --- 2. 定义鼠标点击回调函数 ---
def on_mouse_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    u, v = x, y
    depth_raw = depth[v, u]

    # 检查深度是否有效
    if depth_raw <= 0:
        print(f"像素 ({u}, {v}): 无效深度")
        return

    # 核心公式：反投影
    Z = float(depth_raw) / depth_scale
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    print(f"像素 ({u}, {v}) -> 相机坐标: X={X:.4f}, Y={Y:.4f}, Z={Z:.4f} 米")


# --- 3. 显示图像并绑定鼠标事件 ---
cv2.namedWindow("RGB Image")
cv2.setMouseCallback("RGB Image", on_mouse_click)

print("请在窗口中点击任意像素查看坐标，按 'q' 键退出。")

while True:
    cv2.imshow("RGB Image", rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()