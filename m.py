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
            f"RGB and depth image sizes do not match. "
            f"RGB: {rgb.shape[:2]}, Depth: {depth.shape[:2]}"
        )

    return rgb, depth


def estimate_intrinsics_from_image_size(width, height, fov_deg=90.0):
    """
    Approximate camera intrinsics from image size and horizontal FOV.
    This is only an approximation if true intrinsics are unknown.
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


def convert_depth_to_relative_distance(depth_gray, near=1.0, far=50.0):
    """
    Depth Anything style depth maps are often relative.
    We map white to near and black to far.

    depth_gray: uint8 image where bright = near, dark = far

    Returns:
        Z: approximate distance map
    """
    d = depth_gray.astype(np.float32) / 255.0

    # Invert because white = near, black = far
    # d_inv near 0 => near
    # d_inv near 1 => far
    d_inv = 1.0 - d

    Z = near + d_inv * (far - near)
    return Z


def create_valid_mask(rgb, depth_gray, sky_dark_threshold=20, min_depth_gray=5):
    """
    Build a mask to ignore obvious sky or invalid far regions.

    Heuristic:
    1. Ignore very dark depth pixels
    2. Ignore upper region if it is very dark in depth
    """
    h, w = depth_gray.shape

    valid = np.ones((h, w), dtype=np.uint8)

    # Remove extremely dark pixels
    valid[depth_gray <= min_depth_gray] = 0

    # Stronger removal in upper part, because sky is often there
    upper_limit = int(0.45 * h)
    upper_region = depth_gray[:upper_limit, :]
    sky_like = upper_region < sky_dark_threshold
    valid[:upper_limit, :][sky_like] = 0

    # Optional mild RGB based sky suppression
    # Sky is often bright and bluish, but keep it conservative
    b = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    r = rgb[:, :, 2].astype(np.float32)

    blue_dominant = (b > g + 10) & (b > r + 10) & (b > 100)
    valid[:upper_limit, :][blue_dominant[:upper_limit, :]] = 0

    return valid


def backproject_to_3d(rgb, Z, K, valid_mask, stride=2):
    """
    Back project pixels to 3D in camera coordinates.

    Camera coordinates:
    X to right
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
            colors.append(rgb[v, u, ::-1] / 255.0)  # convert BGR to RGB

    if len(points) == 0:
        raise ValueError("No valid 3D points were generated. Try relaxing the mask thresholds.")

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def rotate_points_for_top_view(points):
    """
    Convert camera frame to a simple world like bird's eye frame.

    Camera frame:
    X right
    Y down
    Z forward

    Bird's eye plane:
    horizontal axis = X
    vertical axis = Z

    We use:
    top_x = X
    top_y = Z

    Also keep height = -Y if needed
    """
    top_x = points[:, 0]
    top_y = points[:, 2]
    height = -points[:, 1]

    return top_x, top_y, height


def rasterize_top_view(top_x, top_y, colors, grid_size=0.10, padding=20):
    """
    Rasterize 3D points onto a top view image.

    grid_size:
        meters per pixel approximately, under relative scaling assumptions

    Since depth is relative, this is only approximate.
    """
    min_x = np.min(top_x)
    max_x = np.max(top_x)
    min_y = np.min(top_y)
    max_y = np.max(top_y)

    width = int(np.ceil((max_x - min_x) / grid_size)) + 2 * padding + 1
    height = int(np.ceil((max_y - min_y) / grid_size)) + 2 * padding + 1

    width = max(width, 200)
    height = max(height, 200)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)

    for i in range(len(top_x)):
        px = int((top_x[i] - min_x) / grid_size) + padding
        py = int((top_y[i] - min_y) / grid_size) + padding

        # invert vertical so farther points go upward in image
        py = height - 1 - py

        if px < 0 or px >= width or py < 0 or py >= height:
            continue

        # Use forward distance as depth in top image
        dist = top_y[i]

        if dist < zbuffer[py, px]:
            zbuffer[py, px] = dist
            canvas[py, px] = (colors[i] * 255).astype(np.uint8)

    return canvas


def densify_top_view(image, kernel_size=3, iterations=2):
    """
    Fill sparse holes for better visualization.
    """
    result = image.copy()

    for _ in range(iterations):
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask = (gray == 0).astype(np.uint8) * 255

        dilated = cv2.dilate(result, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        result[mask > 0] = dilated[mask > 0]

    return result


def save_outputs(output_dir, rgb, depth, valid_mask, top_view_raw, top_view_filled):
    os.makedirs(output_dir, exist_ok=True)

    masked_rgb = rgb.copy()
    masked_rgb[valid_mask == 0] = 0

    cv2.imwrite(os.path.join(output_dir, "01_rgb_input.jpg"), rgb)
    cv2.imwrite(os.path.join(output_dir, "02_depth_input.jpg"), depth)
    cv2.imwrite(os.path.join(output_dir, "03_valid_mask.png"), valid_mask * 255)
    cv2.imwrite(os.path.join(output_dir, "04_masked_rgb.jpg"), masked_rgb)
    cv2.imwrite(os.path.join(output_dir, "05_top_view_raw.jpg"), top_view_raw)
    cv2.imwrite(os.path.join(output_dir, "06_top_view_filled.jpg"), top_view_filled)


def main():
    # =========================================================
    # Change these paths
    # =========================================================
    rgb_path = "front_U1wVHDYpSE7wIUQeItvlhg,41.852282,-87.646483,_input.png"
    depth_path = "front_U1wVHDYpSE7wIUQeItvlhg,41.852282,-87.646483,_raw.png"
    output_dir = "output_birds_eye"

    # =========================================================
    # Tunable parameters
    # =========================================================
    approx_fov_deg = 90.0
    near_dist = 1.0
    far_dist = 10.0
    point_stride = 1
    grid_size = 0.01

    rgb, depth = load_images(rgb_path, depth_path)
    h, w = depth.shape

    print(f"Loaded RGB image:   {rgb_path}")
    print(f"Loaded depth image: {depth_path}")
    print(f"Image size: {w} x {h}")

    K = estimate_intrinsics_from_image_size(w, h, fov_deg=approx_fov_deg)
    Z = convert_depth_to_relative_distance(depth, near=near_dist, far=far_dist)
    valid_mask = create_valid_mask(rgb, depth, sky_dark_threshold=20, min_depth_gray=5)

    points, colors = backproject_to_3d(
        rgb=rgb,
        Z=Z,
        K=K,
        valid_mask=valid_mask,
        stride=point_stride
    )

    print(f"Generated 3D points: {len(points)}")

    top_x, top_y, height_vals = rotate_points_for_top_view(points)

    top_view_raw = rasterize_top_view(
        top_x=top_x,
        top_y=top_y,
        colors=colors,
        grid_size=grid_size,
        padding=20
    )

    top_view_filled = densify_top_view(top_view_raw, kernel_size=3, iterations=3)

    save_outputs(
        output_dir=output_dir,
        rgb=rgb,
        depth=depth,
        valid_mask=valid_mask,
        top_view_raw=top_view_raw,
        top_view_filled=top_view_filled
    )

    print("\nSaved outputs in:", output_dir)
    print("  01_rgb_input.jpg")
    print("  02_depth_input.jpg")
    print("  03_valid_mask.png")
    print("  04_masked_rgb.jpg")
    print("  05_top_view_raw.jpg")
    print("  06_top_view_filled.jpg")


if __name__ == "__main__":
    main()