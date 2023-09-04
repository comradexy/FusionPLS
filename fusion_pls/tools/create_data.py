import argparse
import os
import shutil
import time
import numba
import numpy as np
import imagesize
import datetime
from PIL import Image
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


def pcd_painting(pts, img, Trv2c, P2):
    """Paint points with image.

    Note:
        This function is for KITTI only.

    Args:
        pts (np.ndarray, shape=[N, 3]): Coordinates of points.
        img (np.ndarray, shape=[H, W, 3]): Image.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.

    Returns:
        colors: np.ndarray, shape=[N, 3]: RGB(normalized) of points.
    """
    # Convert points from lidar coordinate to camera coordinate
    pts_cam = lidar_to_camera(pts, Trv2c, P2)
    # Convert points from camera coordinate to image coordinate
    pts_img = pts_cam[:, :2] / pts_cam[:, 2:]
    # Convert image coordinate to pixel coordinates
    pts_img = pts_img.astype(np.int32)
    # Get RGB colors from image
    image = Image.fromarray(img)
    colors = []
    for pt in pts_img:
        x, y = pt
        x = min(max(x, 0), img.shape[1] - 1)
        y = min(max(y, 0), img.shape[0] - 1)
        rgb = image.getpixel((x, y))
        colors.append(rgb)
    colors = np.array(colors)
    # # Normalize RGB
    # colors = colors / 255.0
    return colors


def remove_outside_points(points, Trv2c, P2, image_shape, rect=None):
    """Remove points which are outside of image.

    Note:
        This function is for KITTI only.

    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points.
        rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.
        image_shape (list[int]): Shape of image.

    Returns:
        tuple[np.ndarray]: points, indices
        points: np.ndarray, shape=[N, 3+dims]: Filtered points.
        indices: np.ndarray, shape=[N, 1]: Indices of points which are
            inside of image.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    if rect is not None:
        frustum = camera_to_lidar(frustum.T, Trv2c, r_rect=rect)
    else:
        frustum = camera_to_lidar(frustum.T, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points, indices.squeeze()


def lidar_to_camera(points, velo2cam, P2):
    """Convert points in lidar coordinate to camera coordinate.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in lidar coordinate.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            lidar coordinate to camera coordinate.
        P2 (np.ndarray, shape=[4, 4]): Intrinsics of Camera2.

    Returns:
        np.ndarray, shape=[N, 3]: Points in camera coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if velo2cam.shape != (4, 4):
        velo2cam = np.concatenate(
            [velo2cam, np.array([[0, 0, 0, 1.]], dtype=np.float32)], axis=0)
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    cam_points = points @ velo2cam.T @ P2.T
    return cam_points[..., :3]


def camera_to_lidar(points, velo2cam, r_rect=None):
    """Convert points in camera coordinate to lidar coordinate.

    Note:
        This function is for KITTI only.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if velo2cam.shape != (4, 4):
        velo2cam = np.concatenate(
            [velo2cam, np.array([[0, 0, 0, 1.]], dtype=np.float32)], axis=0)
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    if r_rect is None:
        lidar_points = points @ np.linalg.inv(velo2cam.T)
    else:
        lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """Convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.

    Args:
        corners (np.ndarray): 3d box corners with the shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def projection_matrix_to_CRT_kitti(proj):
    """Split projection matrix of KITTI.

    Note:
        This function is for KITTI only.

    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.

    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.

    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """Get frustum corners in camera coordinates.

    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float, optional): Nearest distance of frustum.
            Defaults to 0.001.
        far_clip (float, optional): Farthest distance of frustum.
            Defaults to 100.

    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def surface_equ_3d(polygon_surfaces):
    """

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - \
                  polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d,
                                     num_surfaces):
    """
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        normal_vec (np.ndarray): Normal vector of polygon_surfaces.
        d (int): Directions of normal vector.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                        points[i, 0] * normal_vec[j, k, 0] +
                        points[i, 1] * normal_vec[j, k, 1] +
                        points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray, optional): Number of surfaces a polygon
            contains shape of (num_polygon). Defaults to None.

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)


def process_sequence(seq, src_path, dst_path, pbar):
    """Process a sequence of KITTI dataset with acceleration."""
    with open(os.path.join(src_path, seq, 'calib.txt'), 'r') as f:
        lines = f.readlines()
    P0 = np.fromstring(lines[0].strip().split(': ')[1], dtype=float, sep=' ').reshape(3, 4)
    P1 = np.fromstring(lines[1].strip().split(': ')[1], dtype=float, sep=' ').reshape(3, 4)
    P2 = np.fromstring(lines[2].strip().split(': ')[1], dtype=float, sep=' ').reshape(3, 4)
    P3 = np.fromstring(lines[3].strip().split(': ')[1], dtype=float, sep=' ').reshape(3, 4)
    Tr = np.fromstring(lines[4].strip().split(': ')[1], dtype=float, sep=' ').reshape(3, 4)

    seq_src_path = os.path.join(src_path, seq)
    seq_dst_path = os.path.join(dst_path, seq)

    seq_velo_src_path = os.path.join(seq_src_path, 'velodyne')
    seq_image_src_path = os.path.join(seq_src_path, 'image_2')
    seq_labels_src_path = os.path.join(seq_src_path, 'labels')

    seq_velo_dst_path = os.path.join(seq_dst_path, 'velodyne')
    seq_vfs_dst_path = os.path.join(seq_dst_path, 'velodyne_fov_single')
    seq_vfm_dst_path = os.path.join(seq_dst_path, 'velodyne_fov_multi')
    seq_ind_dst_path = os.path.join(seq_dst_path, 'indices_fov')
    seq_image_dst_path = os.path.join(seq_dst_path, 'image_2')
    seq_labels_dst_path = os.path.join(seq_dst_path, 'labels_fov')

    os.makedirs(seq_velo_dst_path, exist_ok=True)
    os.makedirs(seq_vfs_dst_path, exist_ok=True)
    os.makedirs(seq_vfm_dst_path, exist_ok=True)
    os.makedirs(seq_image_dst_path, exist_ok=True)
    os.makedirs(seq_ind_dst_path, exist_ok=True)
    os.makedirs(seq_labels_dst_path, exist_ok=True)

    # for frame in tqdm(os.listdir(seq_velo_src_path), desc='Processing seq ' + seq):
    for frame in os.listdir(seq_velo_src_path):
        num = frame.split('.')[0]

        # load points and image
        points = np.memmap(os.path.join(seq_velo_src_path, frame),
                           dtype=np.float32,
                           mode='r').reshape(-1, 4)
        # image = Image.open(os.path.join(seq_image_src_path, num + '.png'))
        # image_size = image.size[::-1]
        image_size = imagesize.get(os.path.join(seq_image_src_path, num + '.png'))[::-1]
        reduced_points, indices = remove_outside_points(points, Tr, P2, list(image_size))

        # save indices to file, dtype=np.bool_
        np.memmap(os.path.join(seq_ind_dst_path, num + '.bin'),
                  dtype=np.bool_,
                  mode='w+',
                  shape=indices.shape)[:] = indices[:]

        # save cropped single-modal frame to file
        np.memmap(os.path.join(seq_vfs_dst_path, num + '.bin'),
                  dtype=np.float32,
                  mode='w+',
                  shape=reduced_points.shape)[:] = reduced_points[:]

        # save cropped multi-modal frame to file
        image = Image.open(os.path.join(seq_image_src_path, num + '.png'))
        image = np.array(image)
        color_pts = pcd_painting(reduced_points[:, :3], image, Tr, P2)
        fused_pts = np.concatenate([reduced_points, color_pts], axis=1)
        np.memmap(os.path.join(seq_vfm_dst_path, num + '.bin'),
                  dtype=np.float32,
                  mode='w+',
                  shape=fused_pts.shape)[:] = fused_pts[:]

        # if dst and src are the same, this step is unnecessary
        if not os.path.samefile(seq_image_src_path, seq_image_dst_path):
            # copy points to dst
            np.memmap(os.path.join(seq_velo_dst_path, num + '.bin'),
                      dtype=np.float32,
                      mode='w+',
                      shape=points.shape)[:] = points[:]

            # copy image to dst
            shutil.copy(os.path.join(seq_image_src_path, num + '.png'),
                        os.path.join(seq_image_dst_path, num + '.png'))

            # copy calib.txt to dst
            calib_src = os.path.join(seq_src_path, 'calib.txt')
            calib_dst = os.path.join(seq_dst_path, 'calib.txt')
            if not os.path.exists(calib_dst):
                with open(calib_dst, 'w') as f:
                    f.write('')
            os.chmod(calib_dst, 0o755)
            shutil.copy(calib_src, calib_dst)

            # copy poses.txt to dst
            poses_src = os.path.join(seq_src_path, 'poses.txt')
            poses_dst = os.path.join(seq_dst_path, 'poses.txt')
            if not os.path.exists(poses_dst):
                with open(poses_dst, 'w') as f:
                    f.write('')
            os.chmod(poses_dst, 0o755)
            shutil.copy(poses_src, poses_dst)

        # if seq < 11, save label
        if int(seq) < 11:
            label = np.memmap(os.path.join(seq_labels_src_path, num + '.label'),
                              dtype=np.int32,
                              mode='r').reshape(-1)
            reduced_label = label[indices]
            # np.savetxt(os.path.join(seq_labels_dst_path, num + '.label'), reduced_label, delimiter=' ', fmt='%d')
            np.memmap(os.path.join(seq_labels_dst_path, num + '.label'),
                      dtype=np.int32,
                      mode='w+',
                      shape=reduced_label.shape)[:] = reduced_label[:]

        pbar.update(1)


def create_cropped_point_cloud(src, dst):
    """Create SemanticKittiF dataset.
    Args:
        src (str): Path to the raw dataset.
        dst (str): Path to save the cropped dataset.
    """
    # print('Creating SemanticKittiF dataset...')
    start_time = time.time()
    src_path = os.path.join(src, 'sequences')
    dst_path = os.path.join(dst, 'sequences')

    sequences = sorted(os.listdir(src_path))
    total = 0
    for seq in sequences:
        total += len(os.listdir(os.path.join(src_path, seq, 'velodyne')))
    pbar = tqdm(total=total, desc='Creating SemanticKittiF dataset')

    for seq in sequences:
        process_sequence(seq, src_path, dst_path, pbar)

    pbar.close()
    end_time = time.time()
    during_time = end_time - start_time
    hh_mm_ss = str(datetime.timedelta(seconds=int(during_time)))
    print('Time cost: {}'.format(hh_mm_ss))
    print('SemanticKittiF dataset created.')


def _create_cropped_point_cloud(src, dst):
    """Create SemanticKittiF dataset with multi-threading.
    Args:
        src (str): Path to the raw dataset.
        dst (str): Path to save the cropped dataset.
    """
    # print('Creating SemanticKittiF dataset...')
    start_time = time.time()
    src_path = os.path.join(src, 'sequences')
    dst_path = os.path.join(dst, 'sequences')

    # Get the list of sequences
    sequences = sorted(os.listdir(src_path))
    total = 0
    for seq in sequences:
        total += len(os.listdir(os.path.join(src_path, seq, 'velodyne')))
    pbar = tqdm(total=total, desc='Creating SemanticKittiF dataset')

    # Create a thread pool
    pool = ThreadPool()

    # Process sequences in parallel
    results = []
    for seq in sequences:
        results.append(
            pool.apply_async(
                process_sequence,
                (seq, src_path, dst_path, pbar)
            )
        )

    # Wait for all threads to complete
    pool.close()
    pool.join()

    pbar.close()
    end_time = time.time()
    during_time = end_time - start_time
    hh_mm_ss = str(datetime.timedelta(seconds=int(during_time)))
    print('Time cost: {}'.format(hh_mm_ss))
    print('SemanticKittiF dataset created.')


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--src', type=str, default='./data/kitti/', help='source path')
parser.add_argument('--dst', type=str, default='./data/kitti_f/', help='destination path')

if __name__ == '__main__':
    args = parser.parse_args()
    src = args.src
    dst = args.dst
    _create_cropped_point_cloud(src, dst)
