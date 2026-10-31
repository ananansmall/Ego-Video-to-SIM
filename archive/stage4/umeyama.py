import numpy as np
from scipy.spatial.transform import Rotation


def umeyama_alignment(src_points, dst_points, with_scale=True):
    """
    Umeyama algorithm for estimating similarity transformation (s, R, t)
    that best aligns src_points to dst_points in the least-squares sense.

    Reference: Umeyama, S. (1991). "Least-squares estimation of transformation
    parameters between two point patterns." IEEE PAMI, 13(4), 376-380.

    Args:
        src_points: (N, 3) numpy array, source points (rendered/mesh points)
        dst_points: (N, 3) numpy array, destination points (real/observed points)
        with_scale: bool, whether to estimate scale (True for similarity, False for rigid)

    Returns:
        s: float, scale factor
        R: (3, 3) numpy array, rotation matrix
        t: (3,) numpy array, translation vector
        T: (4, 4) numpy array, homogeneous transformation matrix
    """
    assert src_points.shape == dst_points.shape, "Point sets must have same shape"
    assert src_points.shape[1] == 3, "Points must be 3D"
    n = src_points.shape[0]
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3), np.eye(4)

    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    src_var = np.mean(np.sum(src_centered ** 2, axis=1))

    sigma = dst_centered.T @ src_centered / n

    U, D_diag, Vt = np.linalg.svd(sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        trace_D = np.sum(D_diag)
        s = trace_D / src_var if src_var > 1e-12 else 1.0
    else:
        s = 1.0

    t = dst_mean - s * R @ src_mean

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    return s, R, t, T


def umeyama_alignment_ransac(src_points, dst_points, with_scale=True,
                              inlier_threshold=0.05, max_iterations=1000,
                              min_inliers=10, confidence=0.99):
    """
    RANSAC-robust Umeyama alignment.

    Args:
        src_points: (N, 3) source points
        dst_points: (N, 3) destination points
        with_scale: whether to estimate scale
        inlier_threshold: distance threshold for inlier classification
        max_iterations: maximum RANSAC iterations
        min_inliers: minimum number of inliers to accept a model
        confidence: RANSAC confidence level for early termination

    Returns:
        best_T: (4, 4) best transformation matrix
        best_inliers: boolean array of inlier flags
        best_s: scale factor
        best_R: rotation matrix
        best_t: translation vector
    """
    n = src_points.shape[0]
    if n < 3:
        T = np.eye(4)
        return T, np.ones(n, dtype=bool), 1.0, np.eye(3), np.zeros(3)

    best_inlier_count = 0
    best_inliers = np.zeros(n, dtype=bool)
    best_T = np.eye(4)
    best_s = 1.0
    best_R = np.eye(3)
    best_t = np.zeros(3)

    sample_size = min(3, n)
    adaptive_max_iter = max_iterations

    for iteration in range(max_iterations):
        if iteration >= adaptive_max_iter:
            break

        indices = np.random.choice(n, sample_size, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        try:
            s, R, t, T = umeyama_alignment(src_sample, dst_sample, with_scale=with_scale)
        except np.linalg.LinAlgError:
            continue

        transformed = (s * R @ src_points.T).T + t
        distances = np.linalg.norm(transformed - dst_points, axis=1)
        inliers = distances < inlier_threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers.copy()
            best_T = T.copy()
            best_s = s
            best_R = R
            best_t = t

            inlier_ratio = inlier_count / n
            if inlier_ratio > 0:
                adaptive_max_iter = min(
                    max_iterations,
                    int(np.log(1 - confidence) / np.log(1 - inlier_ratio ** sample_size)) + 1
                )

    if best_inlier_count >= min_inliers:
        src_inliers = src_points[best_inliers]
        dst_inliers = dst_points[best_inliers]
        try:
            s, R, t, T = umeyama_alignment(src_inliers, dst_inliers, with_scale=with_scale)
            best_T = T
            best_s = s
            best_R = R
            best_t = t
        except np.linalg.LinAlgError:
            pass

    return best_T, best_inliers, best_s, best_R, best_t


def decompose_similarity_transform(T):
    """
    Decompose a 4x4 similarity transformation matrix into (s, R, t).

    Args:
        T: (4, 4) similarity transformation matrix

    Returns:
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    s = np.linalg.norm(T[:3, 0])
    if s < 1e-12:
        return 1.0, np.eye(3), T[:3, 3]
    R = T[:3, :3] / s
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    t = T[:3, 3]
    return s, R, t


def compose_similarity_transform(s, R, t):
    """
    Compose a 4x4 similarity transformation matrix from (s, R, t).

    Args:
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Returns:
        T: (4, 4) similarity transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T
