import os
import cv2
import numpy as np


def load_images(rgb_path, depth_path):
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None:
        raise ValueError(f"Could not read RGB image: {rgb_path}")
    if depth is None:
        raise ValueError(f"Could not read depth image: {depth_path}")

    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"RGB and depth must have same size. "
            f"RGB: {rgb.shape[:2]}, Depth: {depth.shape[:2]}"
        )

    return rgb, depth


def estimate_intrinsics(width, height, fov_deg=90.0):
    """
    Approximate camera intrinsics from image size and horizontal FOV.
    """
    cx = width / 2.0
    cy = height / 2.0
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K


def depth_gray_to_distance(depth_gray, near_dist=1.0, far_dist=50.0):
    """
    Convert grayscale depth to approximate distance.
    Assumes white = near, black = far.
    """
    d = depth_gray.astype(np.float32) / 255.0
    d_inv = 1.0 - d
    Z = near_dist + d_inv * (far_dist - near_dist)
    return Z


def create_valid_mask(rgb, depth_gray, min_depth_gray=5, upper_remove_ratio=0.35):
    """
    Removes obvious invalid / sky-like regions.
    """
    h, w = depth_gray.shape
    valid = np.ones((h, w), dtype=np.uint8)

    valid[depth_gray <= min_depth_gray] = 0

    upper_limit = int(upper_remove_ratio * h)

    # remove top dark region, often sky
    valid[:upper_limit, :][depth_gray[:upper_limit, :] < 20] = 0

    # optional blue-sky suppression
    b = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    r = rgb[:, :, 2].astype(np.float32)

    blue_sky = (b > 100) & (b > g + 10) & (b > r + 10)
    valid[:upper_limit, :][blue_sky[:upper_limit, :]] = 0

    return valid


def backproject_to_point_cloud(rgb, Z, K, valid_mask, stride=1):
    """
    Backproject image pixels into 3D camera coordinates.

    Camera frame:
    X right
    Y down
    Z forward
    """
    h, w = Z.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    points = []
    colors = []

    for v in range(0, h, stride):
        for u in range(0, w, stride):
            if valid_mask[v, u] == 0:
                continue

            z = Z[v, u]
            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])

            # store as RGB
            b, g, r = rgb[v, u]
            colors.append([r, g, b])

    if len(points) == 0:
        raise ValueError("No valid 3D points generated.")

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


def save_point_cloud_ply(points, colors, ply_path):
    """
    Save point cloud as ASCII PLY.
    """
    n = len(points)

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def make_top_down_view(points, colors, grid_size=0.10, padding=20):
    """
    Create a top-down image from the point cloud.

    Uses:
    horizontal axis = X
    vertical axis   = Z

    So the top image is a projection onto the X-Z plane.
    """
    top_x = points[:, 0]
    top_y = points[:, 2]

    min_x = np.min(top_x)
    max_x = np.max(top_x)
    min_y = np.min(top_y)
    max_y = np.max(top_y)

    width = int(np.ceil((max_x - min_x) / grid_size)) + 2 * padding + 1
    height = int(np.ceil((max_y - min_y) / grid_size)) + 2 * padding + 1

    width = max(width, 200)
    height = max(height, 200)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    density = np.zeros((height, width), dtype=np.uint16)

    for i in range(len(points)):
        px = int((top_x[i] - min_x) / grid_size) + padding
        py = int((top_y[i] - min_y) / grid_size) + padding

        # invert y for image coordinates
        py = height - 1 - py

        if 0 <= px < width and 0 <= py < height:
            if density[py, px] == 0:
                canvas[py, px] = colors[i]
            else:
                old_color = canvas[py, px].astype(np.float32)
                new_color = colors[i].astype(np.float32)
                avg_color = ((old_color * density[py, px]) + new_color) / (density[py, px] + 1)
                canvas[py, px] = avg_color.astype(np.uint8)

            density[py, px] += 1

    return canvas, density


def fill_sparse_holes(image, iterations=3, kernel_size=3):
    """
    Simple densification for display.
    """
    result = image.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for _ in range(iterations):
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        holes = (gray == 0).astype(np.uint8) * 255
        dilated = cv2.dilate(result, kernel, iterations=1)
        result[holes > 0] = dilated[holes > 0]

    return result


def save_outputs(output_dir, rgb, depth, valid_mask, top_raw, top_filled):
    os.makedirs(output_dir, exist_ok=True)

    masked_rgb = rgb.copy()
    masked_rgb[valid_mask == 0] = 0

    cv2.imwrite(os.path.join(output_dir, "01_rgb_input.jpg"), rgb)
    cv2.imwrite(os.path.join(output_dir, "02_depth_input.jpg"), depth)
    cv2.imwrite(os.path.join(output_dir, "03_valid_mask.png"), valid_mask * 255)
    cv2.imwrite(os.path.join(output_dir, "04_masked_rgb.jpg"), masked_rgb)
    cv2.imwrite(os.path.join(output_dir, "05_top_down_raw.jpg"), cv2.cvtColor(top_raw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "06_top_down_filled.jpg"), cv2.cvtColor(top_filled, cv2.COLOR_RGB2BGR))


def main():
    # =========================================================
    # Change these file names
    # =========================================================
    rgb_path = "front_U1wVHDYpSE7wIUQeItvlhg,41.852282,-87.646483,_input.png"
    depth_path = "front_U1wVHDYpSE7wIUQeItvlhg,41.852282,-87.646483,_raw.png"
    output_dir = "output_pointcloud_topdown"
    ply_path = os.path.join(output_dir, "point_cloud.ply")

    # =========================================================
    # Tunable parameters
    # =========================================================
    approx_fov_deg = 90.0
    near_dist = 1.0
    far_dist = 50.0
    stride = 1
    grid_size = 0.10

    rgb, depth = load_images(rgb_path, depth_path)
    h, w = depth.shape

    print(f"Loaded RGB image:   {rgb_path}")
    print(f"Loaded depth image: {depth_path}")
    print(f"Image size: {w} x {h}")

    K = estimate_intrinsics(w, h, fov_deg=approx_fov_deg)
    Z = depth_gray_to_distance(depth, near_dist=near_dist, far_dist=far_dist)
    valid_mask = create_valid_mask(rgb, depth, min_depth_gray=5, upper_remove_ratio=0.35)

    points, colors = backproject_to_point_cloud(
        rgb=rgb,
        Z=Z,
        K=K,
        valid_mask=valid_mask,
        stride=stride
    )

    print(f"Generated 3D points: {len(points)}")

    os.makedirs(output_dir, exist_ok=True)
    save_point_cloud_ply(points, colors, ply_path)
    print(f"Saved point cloud: {ply_path}")

    top_raw, density = make_top_down_view(
        points=points,
        colors=colors,
        grid_size=grid_size,
        padding=20
    )

    top_filled = fill_sparse_holes(top_raw, iterations=3, kernel_size=3)

    save_outputs(
        output_dir=output_dir,
        rgb=rgb,
        depth=depth,
        valid_mask=valid_mask,
        top_raw=top_raw,
        top_filled=top_filled
    )

    print("\nSaved outputs in:", output_dir)
    print("  01_rgb_input.jpg")
    print("  02_depth_input.jpg")
    print("  03_valid_mask.png")
    print("  04_masked_rgb.jpg")
    print("  05_top_down_raw.jpg")
    print("  06_top_down_filled.jpg")
    print("  point_cloud.ply")
    print("\nDone.")


if __name__ == "__main__":
    main()