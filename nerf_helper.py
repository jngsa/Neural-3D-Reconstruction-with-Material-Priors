import numpy as np
import torch 
import torch.nn as nn
from torch.nn import functional as F
import imageio
import imageio.v3 as iio
from PIL import Image
import json
import os
import cv2
from pathlib import Path
import time
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')   # cuda_ad_rgb for CUDA / llvm_ad_rgb for GPU
import drjit as dr
import trimesh
import neus_helper as neuh


################################################################################################################################################################
# ScanNet++ Loaders & Helpers
################################################################################################################################################################


# ---------------------------------------------------------------------------
# COLMAP images.txt parsing + convention conversion
# ---------------------------------------------------------------------------
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y],
    ])
 
 
def read_colmap_images(images_txt_path):
    """
    Returns dict: filename -> 4x4 camera-to-world matrix, OpenCV convention
    (x right, y down, z forward -- camera looks down +Z). This frame is
    aligned with the ScanNet++ mesh per the dataset docs.
    """
    poses = {}
    with open(images_txt_path, "r") as f:
        lines = [l for l in f if not l.startswith("#")]
 
    for i in range(0, len(lines), 2):   # pose line + 2D-points line per image
        tokens = lines[i].split()
        qvec = np.array(list(map(float, tokens[1:5])))   # QW QX QY QZ
        tvec = np.array(list(map(float, tokens[5:8])))   # TX TY TZ
        name = tokens[9]
 
        R_w2c = qvec2rotmat(qvec)
        t_w2c = tvec
        R_c2w = R_w2c.T
        C = -R_c2w @ t_w2c
 
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = C
        poses[name] = c2w
 
    return poses
 
 
def colmap_to_opengl(c2w_cv):
    """
    OpenCV local camera axes (x right, y down, z forward) -> OpenGL/Blender/
    NeRF local camera axes (x right, y up, z backward). Flips local Y and Z
    rotation columns only; translation (world position) is unaffected.
    """
    c2w_gl = c2w_cv.copy()
    c2w_gl[:3, 1] *= -1.0
    c2w_gl[:3, 2] *= -1.0
    return c2w_gl
 
 
# ---------------------------------------------------------------------------
# Mask + normalization helpers (unchanged from the original loader)
# ---------------------------------------------------------------------------
def _load_mask(mask_dir, image_filename, h, w):
    """Anonymization mask: 0 = anonymized/invalid, 255 = valid.
    Falls back to an all-valid mask if none exists for this frame."""
    stem = os.path.splitext(os.path.basename(image_filename))[0]
    mask_path = os.path.join(mask_dir, stem + ".png")
    if os.path.exists(mask_path):
        m = np.array(Image.open(mask_path).convert("L"))
        return m.astype(np.float32) / 255.0
    return np.ones((h, w), dtype=np.float32)
 
def _load_depth(depth_dir, image_filename, h, w):
    """
    Load a rendered depth map (uint16 PNG, unit = mm, 0 = invalid).
    Returns depth in **metres** as float32, with 0 for invalid pixels.
    """
    stem = os.path.splitext(os.path.basename(image_filename))[0]
    depth_path = os.path.join(depth_dir, stem + ".png")
    if not os.path.exists(depth_path):
        print(f"  [WARNING] depth not found: {depth_path}")
        return np.zeros((h, w), dtype=np.float32)
    d = np.array(Image.open(depth_path))       # uint16, mm
    if d.shape[:2] != (h, w):                  # safety net, shouldn't trigger if paths are truly aligned
        d = np.array(Image.fromarray(d).resize((w, h), Image.NEAREST))
    return d.astype(np.float32) / 1000.0        # -> metres
 
def compute_scene_normalization(pointcloud_path, camera_pos=None, padding=1.1):
    """
    Compute a center + scale so the scene point cloud fits inside the unit
    bounding box.
    """
    pc = trimesh.load(pointcloud_path, process=False)
    pts = np.asarray(pc.vertices)
 
    if camera_pos is not None:
        pts = np.concatenate([pts, camera_pos], axis=0)
 
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
 
    center = (bbox_min + bbox_max) / 2.0
    half_extent = (bbox_max - bbox_min) / 2.0
    scale = half_extent.max() * padding
 
    return center.astype(np.float32), float(scale)
 
 
# ---------------------------------------------------------------------------
# Main loader -- COLMAP poses, nerfstudio intrinsics, paired by filename
# ---------------------------------------------------------------------------
def load_scannetpp_data(basedir, use_val_every=15, skip_bad=False):
    """
    Args:
        basedir:        path to data/<scene_id>
        use_val_every:  carve out every Nth *train* frame as a val frame
        skip_bad:       drop frames flagged `is_bad` (blur / heavy shadow)
 
    Returns:
        images:      (N,H,W,3) float32 in [0,1]
        depths:      (N,H,W)   float32 metres, normalised, 0=invalid 
        masks:       (N,H,W)   float32 in [0,1]
        poses:       (N,4,4)   camera-to-world, OpenGL/Blender convention,
                                now sourced from COLMAP (mesh-aligned)
        hwf:         [H, W]
        i_split:     [train_idx, val_idx, test_idx]
        norm_center, norm_scale
    """
    dslr_dir = os.path.join(basedir, "dslr")
    img_dir = os.path.join(dslr_dir, "resized_undistorted_images")
    mask_dir = os.path.join(dslr_dir, "resized_undistorted_masks")
    depth_dir = os.path.join(dslr_dir, "render_depth_undistorted")
    json_path = os.path.join(dslr_dir, "nerfstudio", "transforms_undistorted.json")
    colmap_path = os.path.join(dslr_dir, "colmap", "images.txt")
 
    with open(json_path, "r") as f:
        meta = json.load(f)
 
    H, W = int(meta["h"]), int(meta["w"])
    intrinsics = {
        "fl_x": float(meta["fl_x"]),
        "fl_y": float(meta["fl_y"]),
        "cx": float(meta["cx"]),
        "cy": float(meta["cy"]),
    }
 
    # COLMAP poses -- source of truth for camera position/orientation
    colmap_poses_cv = read_colmap_images(colmap_path)
    colmap_poses_gl = {name: colmap_to_opengl(c2w) for name, c2w in colmap_poses_cv.items()}
 
    train_frames = meta["frames"]
    test_frames = meta.get("test_frames", [])
 
    if skip_bad:
        train_frames = [f for f in train_frames if not f.get("is_bad", False)]
 
    dropped = 0
 
    def load_split(frames):
        nonlocal dropped
        imgs, depths, poses, masks, ids = [], [], [], [], []
        for fr in frames:
            fname = fr["file_path"]
            if fname not in colmap_poses_gl:
                dropped += 1
                continue   # keep image/pose lists 1:1 -- skip anything COLMAP doesn't have
 
            fpath = os.path.join(img_dir, os.path.basename(fname))
            img = np.array(Image.open(fpath).convert("RGB")).astype(np.float32) / 255.0
            imgs.append(img)
            depths.append(_load_depth(depth_dir, fname, H, W))
            poses.append(colmap_poses_gl[fname].astype(np.float32))   # <-- COLMAP pose
            masks.append(_load_mask(mask_dir, fname, H, W))
            ids.append(os.path.splitext(os.path.basename(fname))[0])
        return imgs, depths, poses, masks, ids
 
    train_imgs, train_depths, train_poses, train_masks, train_ids = load_split(train_frames)
    test_imgs, test_depths, test_poses, test_masks, test_ids = load_split(test_frames)
 
    if dropped > 0:
        print(f"Warning: {dropped} frame(s) present in transforms_undistorted.json "
              f"had no matching COLMAP pose and were dropped.", flush=True)
 
    all_imgs = train_imgs + test_imgs
    all_masks = train_masks + test_masks
    all_poses = train_poses + test_poses
    all_depths = train_depths + test_depths
    all_ids = train_ids + test_ids
 
    n_train = len(train_imgs)
    n_test = len(test_imgs)
    counts = [0, n_train, n_train + n_test]
    i_split = [np.arange(counts[0], counts[1]),
               np.arange(counts[1], counts[2]),
               np.arange(counts[1], counts[2])]
 
    images = np.stack(all_imgs, axis=0)
    masks = np.stack(all_masks, axis=0)
    poses = np.stack(all_poses, axis=0)
    depths = np.stack(all_depths, axis=0)

    # --- image id based on split ---
    image_ids_split = [
        [all_ids[i] for i in i_split[0]],   # train
        [all_ids[i] for i in i_split[1]],   # validation
        [all_ids[i] for i in i_split[2]]    # test
    ]
 
    # --- scene normalization ---
    pc_path = os.path.join(basedir, "scans", "mesh_aligned_0.05.ply")
    norm_center, norm_scale = compute_scene_normalization(pc_path, camera_pos=poses[:, :3, 3])
    poses[:, :3, 3] = (poses[:, :3, 3] - norm_center) / norm_scale
    depths = depths / norm_scale
 
    ## Debug
    pc = trimesh.load(pc_path, process=False)
    pts = np.asarray(pc.vertices)
    pts_norm = (pts - norm_center) / norm_scale
    print("Point cloud per-axis min:", pts_norm.min(axis=0))
    print("Point cloud per-axis max:", pts_norm.max(axis=0))
    print("Fraction of points outside [-1,1]:",
        np.mean(np.any(np.abs(pts_norm) > 1.0, axis=1)))
 
    cam_norms = np.linalg.norm(poses[:, :3, 3], axis=-1)
    print(f"Cam norm min, max : {cam_norms.min()}, {cam_norms.max()}")

    valid = depths > 0
    if valid.any():
        print(f"Depth     min, max : {depths[valid].min():.4f}, "
            f"{depths[valid].max():.4f}  (normalised metres)")
    else:
        print("  [WARNING] no valid depth pixels loaded at all — check depth_dir / filename matching")
 
    return images, depths, masks, poses, [H, W], intrinsics, i_split, norm_center, norm_scale, image_ids_split


################################################################################################################################################################
# NeRF Basic Functions
################################################################################################################################################################


def load_data(basedir, testskip=1):
    """
    Load and prepare NeRF Synthetic data into desired format for NeRF.

    Args:
        basedir:    Path to scene folder
        testskip:   Use every Nth test/val image
    
    Returns:
        images:         (N, H, W, 3) for N images
        poses:          (N, 4, 4), camera-to-world matrices
        render_poses:   (40, 4, 4) poses for 360 rendering
        hwf:            Height, Width, Focal length
        i_split:        [train_indices, val_indices, test_indices]
    """

    # Load json metadata
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        json_path = os.path.join(basedir, f'transforms_{s}.json')
        with open(json_path, 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    # for nerf synthetic! so that train images = 200
    # splits = ["test", "val", "train"]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        skip = 1 if (s == "train" or testskip == 0) else testskip   # use all images for training, and every Nth for test & val images

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = np.array(Image.open(fname).convert("RGBA"))
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # Split indices for training/validation/test
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)      # N, H, W, 4
    poses = np.concatenate(all_poses, 0)    # N, 4, 4

    # Alpha compositing onto white background
    imgs_alpha = imgs[..., 3]   # raw alpha channel
    imgs = imgs[..., :3] * imgs[..., 3:] + (1. - imgs[..., 3:]) # RGB * alpha + background contribution (white==1.0)

    # Compute focal length
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0)
                             for angle in np.linspace(-180, 180, 41)[:-1]    # 40 frames, drop last
                            ], axis=0)
    
    return imgs, imgs_alpha, poses, render_poses, [H, W, focal], i_split


def pose_spherical(theta, phi, radius):
    """
    Generate a camera-to-world matrix for a camera on a sphere.
    
    Args:
        theta:  Azimuth angle in degrees
        phi:    Elevation angle in degrees (negative = looking down)
        radius: Distance from origin
    
    Returns:
        c2w: (4, 4) camera-to-world matrix
    """
    def rotmat_x(angle):
        angle = np.radians(angle)
        return np.array([
            [1, 0,           0,          0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle),  np.cos(angle), 0],
            [0, 0,           0,          1]
        ], dtype=np.float32)

    def rotmat_y(angle):
        angle = np.radians(angle)
        return np.array([
            [np.cos(angle),  0, np.sin(angle), 0],
            [0,              1, 0,             0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0,              0, 0,             1]
        ], dtype=np.float32)

    # Start with camera at [0,0,radius] looking toward origin
    # then rotate: first tilt up/down (phi), then spin left/right (theta)
    c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    c2w = rotmat_x(phi) @ c2w
    c2w = rotmat_y(theta) @ c2w

    # Blender uses a different axis convention: flip y and z axes
    # This converts from Blender camera space to NeRF world space
    c2w = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float32) @ c2w

    return c2w


def positional_encode(xcoords, raydirs, L_pos, L_dir):
    """
    Carry out positional encoding for all position coordinate values and all ray directions.
    Implemented based on Equation 4, Section 5.1 in NeRF paper.

    Args:
        xcoords:    Coordinate values of position x
        raydirs:    Ray directions
        L_pos:      Number of encoding functions for positions
        L_dir:      Number of encoding functions for viewing directions
        
    Returns:
        Positional encoding of coordinates and ray directions respectively.
    """

    xcoords_encoded = [xcoords]
    for l in range(L_pos):
        xcoords_encoded.append(torch.sin((2 ** l) * torch.pi * xcoords))    # sin in Equation 4, Section 5.1
        xcoords_encoded.append(torch.cos((2 ** l) * torch.pi * xcoords))    # cos in Equation 4, Section 5.1

    xcoords_encoded = torch.cat(xcoords_encoded, dim=-1)    # join all positional encodings together for each point

    raydirs = raydirs / torch.linalg.norm(raydirs, dim=-1, ord=2, keepdim=True)   # normalize ray direction vectors
    raydirs_encoded = [raydirs]
    for l in range(L_dir):
        raydirs_encoded.append(torch.sin((2**l) * torch.pi * raydirs))
        raydirs_encoded.append(torch.cos((2**l) * torch.pi * raydirs))

    raydirs_encoded = torch.cat(raydirs_encoded, dim=-1)

    return xcoords_encoded, raydirs_encoded


def get_coarse_samples(raydirs, N_c, t_bin_leftbound, t_gap, rayorig):
    """
    Carry out 'coarse' stratified sampling at N_c locations along the ray.

    Args:
        raydirs:        Ray directions 
        N_c:            Number of sampled locations per ray
        t_bin_leftbound:Left boundaries of evenly spaced bins    
        t_gap:          Bin width
        rayorig:        Ray origins
    
    Returns:
        Tuple of chosen points along the ray, and sampled depths.
    """

    rand_vals = torch.rand(*list(raydirs.shape[:2]) + [N_c]).to(raydirs)    # Values sampled randomly from uniform distribution
    t_sampled = t_bin_leftbound + rand_vals * t_gap     # in the paper t represents depth
    
    # r(t) = o + td equation in paper
    #
    #  Indexing : H = height of ray batches, W = width of ray batches
    #   --> rayorig:    (H, W, 3)   ==> (H, W, 1, 3)
    #   --> t_sampled:  (H, W, N_c) ==> (H, W, N_c, 1)
    #   --> raydirs:    (H, W, 3)   ==> (H, W, 1, 3)
    #   --> t_sampled * raydirs     ==> (H, W, N_c, 3)
    #   --> r_sampled               ==> (H, W, N_c, 3)
    #
    r_sampled = rayorig[..., None, :] + t_sampled[..., :, None] * raydirs[..., None, :]    # r represents points along a ray

    return (r_sampled, t_sampled)


def get_coarse_samples_box(raydirs, rayorig, N_total, bound_min=-1.0, bound_max=1.0):
    flat_dirs = raydirs.reshape(-1, 3)
    flat_orig = rayorig.reshape(-1, 3)
    near, far = neuh.get_near_far_bounds_box(flat_orig, flat_dirs, bound_min, bound_max)  # (M,1) each

    u = torch.rand(flat_dirs.shape[0], N_total, device=raydirs.device)
    t_sampled_flat = near + (far - near) * u        # stratified within each ray's own bounds
    t_sampled_flat, _ = torch.sort(t_sampled_flat, dim=-1)

    r_sampled_flat = flat_orig[:, None, :] + t_sampled_flat[..., :, None] * flat_dirs[:, None, :]

    r_sampled = r_sampled_flat.reshape(*raydirs.shape[:-1], N_total, 3)
    t_sampled = t_sampled_flat.reshape(*raydirs.shape[:-1], N_total)
    return r_sampled, t_sampled


def get_fine_samples(coarse_weights, N_f, coarse_t, t_f, rayorig, raydirs):
    """
    Carry out 'fine' sampling, where samples are biased towards relevant parts of the volume.
    Based on Equation 5 in Section 5.2 and the concept of inverse transform sampling.

    Args:
        coarse_weights: weights for the colours along the rays resulting from 'coarse' network
        N_f:            Number of sampled locations per ray
        coarse_t:       Sampled depths from coarse sampling
        t_f:            Far bound for sampling
        rayorig:        Ray origins
        raydirs:        Ray directions
    
    Returns:
        Tuple of chosen points along the ray, and sampled depths.
    """

    # Obtain probability density functions (PDFs) and cumulative distribution functions (CDFs) from coarse weights
    coarse_weights = coarse_weights + 1e-5 # avoid zero division
    pdfs = coarse_weights / torch.sum(coarse_weights, dim=-1, keepdim=True)
    cdfs = torch.cumsum(pdfs, dim=-1)
    cdfs = torch.cat([torch.zeros_like(cdfs[...,:1]), cdfs[...,:-1]], dim=-1)   # shift cdf to the right to get lower bounds, e.g. [0.3, 0.6, 1.0] --> [0.0, 0.3, 0.6]

    # Obtain uniform samples
    samples = torch.rand(list(cdfs.shape[:-1]) + [N_f]).to(coarse_weights)  # generate random numbers of shape (H, W, N_f)
    
    # Inverse transform sampling : Randomly generate a number u in [0,1], then apply inverse cumulative distribution F^-1(u) to get a random sample
    idxs = torch.searchsorted(cdfs, samples, right=True)
    #
    # Get samples' lower boundaries
    t_lower_bound = torch.gather(coarse_t, dim=2, index=idxs - 1)
    #
    # Cap maximum index 
    capped_idxs = idxs.clone()
    max_idx = cdfs.shape[-1]
    capped_idxs = torch.where(idxs == max_idx, max_idx - 1, idxs)
    #
    # Get samples' upper boundaries
    t_upper_bound = torch.gather(coarse_t, dim=2, index=capped_idxs)
    mask = (idxs == max_idx)
    t_upper_bound = torch.where(mask, 
                                torch.full_like(t_upper_bound, t_f), # set upper bound of last bin to farthest sampling bound
                                t_upper_bound)      
    #
    # Get gap between bins
    t_gaps = t_upper_bound - t_lower_bound
    #
    # Random sampling within new bins
    new_samples = torch.rand_like(t_gaps).to(rayorig)
    t_sampled = t_lower_bound + new_samples * t_gaps    # new fine depth samples

    # Combine coarse and fine samplings (sorted)
    t_sampled, _ = torch.sort(torch.cat([coarse_t, t_sampled.detach()], dim=-1), dim=-1)

    # Calculate chosen points along ray using  r(t) = o + td equation in paper
    r_sampled = rayorig[..., None, :] + t_sampled[..., :, None] * raydirs[..., None, :]

    return (r_sampled, t_sampled)


def render_volume(r, raydirs, chunk_size, model, t, lambda_val=None):
    """
    By using the neural network model (model), predict colours (c) and volume densities (sigma) for all 3D points (r)
    along each rays given the viewing direction (raydirs) of rays.
    Implemented based on Section 3 and Fig. 7 in Supplementary of NeRF paper.
    
    Args:
        r:          3D points along a ray, shape: (H, W, N_c/N_f, 3)
        raydirs:    Ray directions, shape: (H, W, 3)
        chunk_size: Max no. of points to process at one go, for memory management purposes
        model:      Neural network model
        t:          Sampled depths
        lambda_val: Value that determines the current grid resolution to be trained at. (Part of coarse-to-fine optimization)

    Returns:
        C_r :       Predicted pixel colours for each rays
        weights:    Weights for the colours along the rays
    """
    
    # Flatten tensors in order to feed into model
    r_flat = r.reshape((-1,3))
    raydirs_repeated = raydirs.unsqueeze(2).repeat(1, 1, r.shape[-2], 1)  # repeat ray direction for every sampled 3D points
    raydirs_flat = raydirs_repeated.reshape((-1,3))

    # Accumulate results (colour, volume density)
    c = []
    sigma = []

    # Predict colour and volume density for each point using model by chunks
    for chunk in range(0, r_flat.shape[0], chunk_size):
        # Extract points & directions per chunk
        r_chunk = r_flat[chunk : chunk + chunk_size]
        raydirs_chunk = raydirs_flat[chunk : chunk + chunk_size]
        preds = model(r_chunk, raydirs_chunk)
        c.append(preds["c"])
        sigma.append(preds["sigma"])

    # Reshape predictions back to original shape
    c = torch.cat(c).reshape(r.shape)               # shape: H, W, N_c, RGB     || c[i,j,k] = RGB value at sample k of ray(i,j)
    sigma = torch.cat(sigma).reshape(r.shape[:-1])  # shape: H, W, N_c          || sigma[i,j,k] = Volume density at sample k of ray(i,j) 

    sigma = torch.clamp(sigma, min=0.0, max=50.0)
    c = torch.clamp(c, min=0.0, max=1.0)
    
    # Check for NaN and replace with zeros
    if torch.isnan(sigma).any():
        sigma = torch.nan_to_num(sigma, nan=0.0)
    if torch.isnan(c).any():
        c = torch.nan_to_num(c, nan=0.5)  # Gray for NaN colors

    # Calculate distances (delta) between sample points along the ray
    # Based on text right after Equation 3 in Section 4
    delta = t[..., 1:] - t[..., :-1]    # t[all except 1st] - t[all except last] = difference between one t and its next 
    delta = torch.clamp(delta, min=1e-6)

    # alpha values represent opacity, 0:transparent, 1:opaque.
    # Add final distance of Infinity to ensure alpha value will always be 1 at the end, recovers the distance value lost due to subtraction
    final_dist = torch.full(delta[..., :1].shape, 1e3, device=delta.device, dtype=delta.dtype)
    delta = torch.cat([delta, final_dist], dim=-1)

    # Scale distances by ray direction magnitude
    delta = delta * raydirs.norm(dim=-1).unsqueeze(-1)

    # Calculate alphas of 3D points using volume densities (sigma) and distance between points (delta).
    # Alpha aka probability that a ray hitting this particle will be absorbed/scattered.
    # Based on text right after Equation 3 in Section 4
    alpha = 1.0 - torch.exp(- sigma * delta)
    alpha = torch.clamp(alpha, 0.0, 1.0)

    # Calculate accumulated transmittance (T) along each rays.
    # Accumulated transmittance aka probability that a ray will travel from t_n to (excluding) t_i without hitting any particles.
    # Based on T_i from Equation 3 in Section 4, not exactly but mathematically similar.
    #
    # Paper introduces T_i = exp(- sum from (j=1) to (i-1) for (delta_j * sigma_j)).
    # In this implementation, 
    #   we know 1.0 - alpha == exp(sigma * delta),
    #   we use T_i = product from (j=1) to (i-1) for (1.0 - alpha_j) = product from (j=1) to (i-1) for (exp(sigma_j * delta_j))
    #   we also know that exp of sums are mathematically equal to product of exponents
    # --> thus T_i = exp(- sum from (j=1) to (i-1) for (delta_j * sigma_j))
    #
    # Transmittance at the point i
    log_one_minus_alpha = torch.log(1.0 - alpha + 1e-10)
    log_T = torch.cumsum(log_one_minus_alpha, dim=-1)
    log_T = torch.cat([torch.zeros_like(log_T[..., :1]), log_T[..., :-1]], dim=-1)
    T = torch.exp(log_T)
    T = torch.clamp(T, 0.0, 1.0)
    # # Gets transmittance before reaching the point i
    # ones_init = torch.ones_like(T[..., :1])
    # T = torch.cat([ones_init, T[..., :-1]], dim=-1)
    # T = torch.clamp(T, 0.0, 1.0)

    # Calculate weights for the colours along the rays using transmittance and alpha.
    # Based on equation of weights w_i in Equation 5 in Section 5.2
    weights = T * alpha
    weights = torch.clamp(weights, 0.0, 1.0) 
    
    # Calculate pixel colours (C_r) for the rays as a weighted sum of all sampled colours (c) along each ray.
    # Based on Equation 5 in Section 5.2
    #
    # weights.shape :   (H, W, N_c) == reshape ==> (H, W, N_c, 1)
    # c.shape :         (H, W, N_c, RGB)
    # before sum :      (H, W, N_c, RGB)
    # C_r :             (H, W, RGB) <-- summed along N_c
    #
    C_r = (weights[..., None] * c).sum(dim=-2) 
    C_r = torch.clamp(C_r, 0.0, 1.0) 
    
    return C_r, weights   


def run_nerf_once(raydirs, N_c, t_bin_leftbound, t_gap, rayorig, chunk_size, model_c, N_f, t_f, model_f):
    """
    Run through NeRF for one iteration.
    ** Naming convention => _c : coarse, _f : fine

    Returns:
        Expected colour using coarse and fine network respectively.
    """
    
    # Coarse network
    r_sampled_c, t_sampled_c = get_coarse_samples(raydirs, N_c, t_bin_leftbound, t_gap, rayorig)
    C_r_c, weights_c = render_volume(r_sampled_c, raydirs, chunk_size, model_c, t_sampled_c)

    # Fine network
    r_sampled_f, t_sampled_f = get_fine_samples(weights_c, N_f, t_sampled_c, t_f, rayorig, raydirs)
    C_r_f, _ = render_volume(r_sampled_f, raydirs, chunk_size, model_f, t_sampled_f)

    return C_r_c, C_r_f


def run_ingpnerf_once(raydirs, N, rayorig, chunk_size, model, lambda_val=None):
    """
    Run through NeRF for one iteration based on INGP model.

    Returns:
        Expected colour, weights, sampled depth
    """
    if torch.isnan(raydirs).any() or torch.isinf(raydirs).any():
        print("WARNING: NaN/Inf in raydirs!")
        raydirs = torch.nan_to_num(raydirs, nan=0.0)
    if torch.isnan(rayorig).any() or torch.isinf(rayorig).any():
        print("WARNING: NaN/Inf in rayorig!")
        rayorig = torch.nan_to_num(rayorig, nan=0.0)
    
    # Single pass, double the samples
    N_total = N * 2
    r_sampled, t_sampled = get_coarse_samples_box(raydirs, rayorig, N_total)

    C_r, weights = render_volume(r_sampled, raydirs, chunk_size, model, t_sampled, lambda_val=lambda_val)

    return C_r, weights, t_sampled


def compute_batch_near_far(rayorig, raydir, bound_min=-1.0, bound_max=1.0):
    """
    Derive a single scalar (near, far) for a whole batch of rays by intersecting
    them with the normalized scene bounding box. Vanilla NeRF's get_coarse_samples
    wants one global t_bin_leftbound / t_gap rather than per-ray bounds, so we
    take the min near / max far across the batch -- wide enough to cover every
    ray, recomputed fresh per training step to stay adaptive to whichever image
    was sampled.
    """
    flat_o = rayorig.reshape(-1, 3)
    flat_d = raydir.reshape(-1, 3)
    near, far = neuh.get_near_far_bounds_box(flat_o, flat_d, bound_min, bound_max)
    near = near[torch.isfinite(near)]
    far = far[torch.isfinite(far)]
    if near.numel() == 0 or far.numel() == 0:
        # ray batch missed the box entirely (shouldn't normally happen) -- fall back
        return 0.0, 2.5
    return float(near.min().item()), float(far.max().item())


def evaluate_nerf(model_c, model_f, eval_images, eval_poses, init_raydir,
                   N_c, N_f, chunk_size, device, loss_fn, max_images=None, row_chunk=32):
    """
    Evaluate coarse+fine vanilla NeRF on a set of images and poses.
    Processes each image in row blocks to bound peak memory, since a full
    high-res image at N_c+N_f samples/ray otherwise materializes multi-GB
    tensors in render_volume.
    """
    model_c.eval()
    model_f.eval()

    height, width = init_raydir.shape[:2]
    losses = []
    psnrs = []

    n_eval = len(eval_images) if max_images is None else min(max_images, len(eval_images))
    random_pick = np.random.randint(0, high=n_eval)
    C_r_return = None

    with torch.no_grad():
        for i in range(n_eval):
            c2w = torch.FloatTensor(eval_poses[i]).to(device)
            rot = c2w[:3, :3]
            t = c2w[:3, 3]

            raydir_full = torch.einsum("ij,hwj->hwi", rot, init_raydir)
            rayorig_full = t.expand(raydir_full.shape)
            target_full = torch.FloatTensor(eval_images[i]).to(device)

            sum_sq_c = 0.0
            sum_sq_f = 0.0
            n_vals = 0
            C_r_f_full = torch.zeros_like(target_full) if i == random_pick else None

            for r0 in range(0, height, row_chunk):
                r1 = min(r0 + row_chunk, height)
                raydir_block = raydir_full[r0:r1]
                rayorig_block = rayorig_full[r0:r1]
                target_block = target_full[r0:r1]

                t_near, t_far = compute_batch_near_far(rayorig_block, raydir_block)
                t_gap = (t_far - t_near) / N_c

                C_r_c, C_r_f = run_nerf_once(raydir_block, N_c, t_near, t_gap, rayorig_block,
                                              chunk_size, model_c, N_f, t_far, model_f)

                sum_sq_c += F.mse_loss(C_r_c, target_block, reduction='sum').item()
                sum_sq_f += F.mse_loss(C_r_f, target_block, reduction='sum').item()
                n_vals += target_block.numel()

                if C_r_f_full is not None:
                    C_r_f_full[r0:r1] = C_r_f

            l_c = sum_sq_c / n_vals
            l_f = sum_sq_f / n_vals
            loss = l_c + l_f
            psnr = 10.0 * np.log10(1.0 / l_f)

            losses.append(loss)
            psnrs.append(psnr)

            if i == random_pick:
                C_r_return = C_r_f_full

    return np.mean(losses), np.mean(psnrs), C_r_return


def evaluate_ingp(model, eval_images, eval_poses, init_raydir,
             N, chunk_size, device, loss_fn,
             max_images=None, lambda_val=None):
    """
    Evaluate model on a set of images and poses based on INGP model.
    Returns average loss and PSNR across all evaluated images.
    """
    model.eval()

    losses = []
    psnrs  = []

    n_eval = len(eval_images) if max_images is None else min(max_images, len(eval_images))
    random_pick = np.random.randint(0, high=n_eval)
    C_r_return = None

    with torch.no_grad():
        for i in range(n_eval):
            # Get pose for this image
            c2w = torch.FloatTensor(eval_poses[i]).to(device)
            rot = c2w[:3, :3]
            t   = c2w[:3,  3]

            # Build rays
            raydir  = torch.einsum("ij,hwj->hwi", rot, init_raydir)
            rayorig = t.expand(raydir.shape)

            # Render full image
            C_r, weights, t_sampled = run_ingpnerf_once(
                raydir, N, rayorig,
                chunk_size, model, lambda_val=lambda_val
            )

            w_flat = weights.reshape(-1, N * 2)
            t_flat = t_sampled.reshape(-1, N * 2)

            target = torch.FloatTensor(eval_images[i]).to(device)
            l_recon = loss_fn(C_r, target)
            l_distort = distortion_loss(w_flat, t_flat) * 1e-3
            l_sparse = sparsity_loss(w_flat) * 1e-4
            loss = l_recon + l_distort + l_sparse
            psnr   = 10.0 * torch.log10(1.0 / l_recon)

            losses.append(loss.item())
            psnrs.append(psnr.item())

            if i == random_pick:
                C_r_return = C_r

    return np.mean(losses), np.mean(psnrs), C_r_return


def spherify_poses(n_frames, cam_distance, elevation_deg=30.0):
    """
    Generate camera poses on a sphere, mimicking the original NeRF spherify approach.
    Cameras all point toward the origin from positions on a sphere.
    """
    poses = []
    elevation = np.radians(elevation_deg)
    
    for i in range(n_frames):
        azimuth = 2 * np.pi * i / n_frames
        
        # Camera position on sphere
        cam_pos = np.array([
            cam_distance * np.cos(elevation) * np.sin(azimuth),
            cam_distance * np.cos(elevation) * np.cos(azimuth),
            cam_distance * np.sin(elevation),
        ])
        
        # z-axis: camera points toward origin
        z_axis = -cam_pos / np.linalg.norm(cam_pos)
        
        # x-axis: right vector, perpendicular to z and world up
        up = np.array([0.06618502, -0.01944642, 0.99761784])
        up = up / np.linalg.norm(up)
        x_axis = np.cross(z_axis, up)
        x_axis /= np.linalg.norm(x_axis)
        
        # y-axis: recomputed up, perpendicular to both
        y_axis = np.cross(x_axis, z_axis)
        y_axis /= np.linalg.norm(y_axis)
        
        # Build c2w (camera-to-world) rotation matrix
        # columns: right, up, -forward (standard OpenGL/NeRF convention)
        rot = np.stack([x_axis, y_axis, -z_axis], axis=-1)  # shape (3,3)
        poses.append(rot)
    
    return poses


def render_spherify_video(model, init_raydir, init_rayorig,
                          N, t_bin_leftbound, t_gap,
                          chunk_size, cam_distance, device,
                          n_frames=120, elevation_deg=30.0,
                          output_path="result/ingpnerfactor_spherify-lego.mp4", fps=30, lambda_val=None):
    """
    Render a spherified 360-degree video of NeRF object.
    """
    model.eval()
    
    frames = []
    poses = spherify_poses(n_frames, cam_distance, elevation_deg)
    
    print(f"Rendering {n_frames} frames at elevation {elevation_deg}°...", flush=True)
    
    with torch.no_grad():
        for i, rot in enumerate(poses):
            rot_tensor = torch.FloatTensor(rot).to(device)
            
            # Apply rotation to ray directions and origins — same as your test view setup
            raydir  = torch.einsum("ij,hwj->hwi", rot_tensor, init_raydir)
            rayorig = (rot_tensor @ init_rayorig).expand(raydir.shape)
            
            C_r, _, _ = run_ingpnerf_once(
                raydir, N, t_bin_leftbound, t_gap, rayorig,
                chunk_size, model, lambda_val=lambda_val
                )
            
            frame = np.ascontiguousarray((C_r.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            frames.append(frame)
            
            if i % 10 == 0:
                print(f"  Frame {i+1}/{n_frames}", flush=True)
    
    # Stack frames into a single 4D array (N, H, W, C)
    print(f"Stacking {len(frames)} frames...", flush=True)
    frames_array = np.stack(frames, axis=0)
    print(f"Frames array shape: {frames_array.shape}", flush=True)

    print(f"Saving to {output_path}...", flush=True)
    with imageio.get_writer(output_path, fps=fps, quality=8, format="FFMPEG", codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)
    
    print("Done!", flush=True)
    
    return frames


def distortion_loss(weights, t_vals, chunk_size=1024):
    """
    Approximate distortion regularizer (Mip-NeRF 360, Eq. 15, simplified).
    Penalizes weight mass that is spread out along a ray instead of concentrated at a single depth.
 
    Args:
        weights: (N, S) per-sample weights from render_volume
        t_vals:  (N, S) per-sample depths
 
    Returns:
        scalar loss
    """
    N, S = weights.shape  # N rays, S samples
    device = weights.device
    
    # Prepare tensors for chunked computation
    total_loss = torch.zeros(N, device=device)
    
    # Process rays in chunks to reduce memory
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        w_chunk = weights[start:end]  # (chunk, S)
        t_chunk = t_vals[start:end]  # (chunk, S)
        
        # Compute pairwise differences efficiently
        # Using broadcasting but limited to chunk size
        # t_chunk: (C, S, 1) - (C, 1, S) -> (C, S, S)
        t_diff = t_chunk.unsqueeze(-1) - t_chunk.unsqueeze(-2)  # (C, S, S)
        
        # Pairwise weights product
        w_prod = w_chunk.unsqueeze(-1) * w_chunk.unsqueeze(-2)  # (C, S, S)
        
        # Combine and sum
        loss_chunk = (w_prod * torch.abs(t_diff)).sum(dim=(-1, -2))  # (C,)
        total_loss[start:end] = loss_chunk
    
    return total_loss.mean()
 
 
def sparsity_loss(weights):
    """L1 penalty on per-sample weights — discourages diffuse, low-level density."""
    return weights.abs().mean()


SCENE_BOUNDS = ([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])

def normalize_coords(coords, scene_bounds=SCENE_BOUNDS):
    """
    Normalize coordinates to [0, 1] range for hash grid encoding.
    
    Args:
        coords: Input coordinates
        scene_bounds: Tuple of (min_bound, max_bound) for the scene
    """
    min_bound = torch.tensor(scene_bounds[0], device=coords.device)
    max_bound = torch.tensor(scene_bounds[1], device=coords.device)
    
    # Normalize to [0, 1]
    normalized = (coords - min_bound) / (max_bound - min_bound)
    normalized = normalized.clamp(0.0,1.0)
    return normalized
 
def in_scene_bounds(pts, bounds=SCENE_BOUNDS):
    lo = torch.tensor(bounds[0], device=pts.device)
    hi = torch.tensor(bounds[1], device=pts.device)
    return ((pts >= lo) & (pts <= hi)).all(dim=-1)


################################################################################################################################################################
# NeRFactor Helper Functions
################################################################################################################################################################

def l2_normalize(v, dim=-1, eps=1e-6):
    return v / torch.sqrt(torch.sum(v ** 2, dim=dim, keepdim=True) + eps)

# Compute analytic surface normals, n_a
def compute_analytic_normals(model, xcoords, eps=0.05, scene_bounds=None):
    """
    Get analytical surface normals of INGPNeRF's density field by computing finite differences (floating point 32 instead of floating point 16 limited by tcnn).
    """

    def sigma_at(pts):
        norm_pts = normalize_coords(pts, scene_bounds)
        pos_encoding = model.pos_encoder(norm_pts)
        density_output = model.density_mlp(pos_encoding)
        return F.softplus(density_output[:,0].float()).clamp(max=40.0)  # cast to fp32

    def get_grad(pts):
        ex = torch.zeros_like(pts)
        ex[:,0] = eps
        ey = torch.zeros_like(pts)
        ey[:,1] = eps
        ez = torch.zeros_like(pts)
        ez[:,2] = eps

        gx = sigma_at(pts + ex) - sigma_at(pts - ex) # central difference
        gy = sigma_at(pts + ey) - sigma_at(pts - ey)
        gz = sigma_at(pts + ez) - sigma_at(pts - ez)

        return torch.stack([gx, gy, gz], dim=-1)   # shape (N,3)
    
    with torch.no_grad():
        grad = get_grad(xcoords)

    grad_norm = torch.norm(grad, dim=-1, keepdim=True)
    normals = torch.where(
        grad_norm > 1e-3,
        -grad / grad_norm.clamp(min=1e-3),
        torch.zeros_like(grad)  # mark as invalid
    )

    return normals


def compute_analytic_normals_vanilla(model, xcoords, eps=0.05, scene_bounds=None):
    """
    Compute analytic surface normals from Vanilla NeRF's density field.
    Uses finite differences with gradient checkpointing for memory efficiency.
    """
    def sigma_at(pts):
        """Query density at points using vanilla NeRF."""
        with torch.no_grad():
            # Vanilla NeRF doesn't need normalization if using positional encoding
            dummy_dirs = torch.zeros_like(pts)
            dummy_dirs[...,2] = 1.0
            result = model(pts, dummy_dirs)  # Dummy ray directions
            return result["sigma"].float()

    def get_grad(pts):
        """Compute gradient via central differences."""
        ex = torch.zeros_like(pts)
        ex[:, 0] = eps
        ey = torch.zeros_like(pts)
        ey[:, 1] = eps
        ez = torch.zeros_like(pts)
        ez[:, 2] = eps

        gx = sigma_at(pts + ex) - sigma_at(pts - ex)
        gy = sigma_at(pts + ey) - sigma_at(pts - ey)
        gz = sigma_at(pts + ez) - sigma_at(pts - ez)

        return torch.stack([gx, gy, gz], dim=-1)

    with torch.no_grad():
        grad = get_grad(xcoords)

    # Normalize and point inward (negative gradient direction)
    # normals = -F.normalize(grad, dim=-1, eps=1e-8)
    grad_norm = torch.norm(grad, dim=-1, keepdim=True)
    normals = torch.where(
        grad_norm > 1e-3,
        -grad / grad_norm.clamp(min=1e-3),
        torch.zeros_like(grad)  # mark as invalid
    )
    return normals


# Extract surface points, x_surf
# Location at which ray is expected to terminate based on sigma
def compute_surface_pts(ray_dirs, ray_origs, t_samples, weights):
    """
    Compute expected surface points from volume-rendering weights.
    """

    ## Peak depth
    peak_idx = weights.argmax(dim=-1) # shape (N,)
    peak_depth = t_samples.gather(1, peak_idx.unsqueeze(-1)).squeeze(-1)

    ## Expected depth
    # expected_depth = (weights * t_samples).sum(dim=-1)  # shape (N,)
    # total_weight = weights.sum(dim=-1)
    # expected_depth = expected_depth / (total_weight + 1e-8)

    # x_surf 
    surface_pts = ray_origs + peak_depth.unsqueeze(-1) * ray_dirs
    return surface_pts


# Generate light directions and solid angles for Lighting
def get_lights_and_angles(height=16):
    """
    Creates light directoins and solid angles for a latitude-longitude environment map.

    Resolution: H x 2H (based on paper 16 x 32)

    Returns:
        l_xyz: (L,3) unit directions toward each light pixel, ω_i
        areas: (L,1) solid angle Δω_i for each light pixel
    """

    width = 2 * height

    # Lat-long grid centers
    theta = torch.linspace(0, np.pi, height+1)[:-1] + 0.5 * np.pi / height
    phi = torch.linspace(0, 2*np.pi, width+1)[:-1] + np.pi / width
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    theta = theta.reshape(-1) # shape (L,)
    phi = phi.reshape(-1)

    # Unit direction vectors
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    l_xyz = torch.stack([x,y,z], dim=-1)    # shape (L,3)

    # Solid angle
    dtheta = np.pi / height
    dphi = 2 * np.pi / width
    areas = torch.sin(theta) * dtheta * dphi
    areas = areas.unsqueeze(-1) # shape (L,1)

    return l_xyz, areas


def cooktorrance_brdf(albedo, metallic, roughness, normals, w_i, w_o):
    """
    Cook-Torrance microfacet BRDF

    f_r = f_diffuse + f_specular

    Args:
        albedo:     (N,3) base colour in [0,1]
        metallic:   (N,1) metallic factor in [0,1]
        roughness:  (N,1) roughness facton in [0.04,1]
        normals:    (N,1,3) surface normals
        w_i:        (1,L,3) incident light directions
        w_o:        (N,1,3) outgoing view directions

    Returns:
        f_r:        (N,L,3) BRDF values for every surface point x every light combination 
    """

    eps = 1e-4

    # Expand params to (N,1,...)
    alb = albedo.unsqueeze(1)
    met = metallic.unsqueeze(1)
    rou = roughness.unsqueeze(1).clamp(min=0.04, max=1.0)

    # Half vector, (N,L,3)
    h = F.normalize(w_i + w_o, dim=-1)

    # Dot products
    n_dot_wi = (normals * w_i).sum(-1).clamp(min=eps)   # (N,L)
    n_dot_wo = (normals * w_o).sum(-1).clamp(min=eps)   # (N,L)
    n_dot_h = (normals * h).sum(-1).clamp(min=eps)   # (N,L)
    wo_dot_h = (w_o * h).sum(-1).clamp(min=eps)   # (N,L)

    # GGX roughness
    alpha = rou.squeeze(-1) ** 2   # (N,1)
    alpha2 = alpha ** 2            # (N,1)

    # D : Normal Distribution Function
    ndh2 = n_dot_h ** 2     # (N,L)
    denom_D = (torch.pi * ((ndh2 * (alpha2 - 1.0) + 1.0)**2)).clamp(min=eps) # (N,L)
    D = (alpha2 / denom_D).clamp(max=100.0)

    # G : Geometry Function
    k = (alpha / 2.0).clamp(min=eps)         # (N,1)
    G_wi = n_dot_wi / ((n_dot_wi * (1.0 - k)) + k + eps) # (N,L)
    G_wo = n_dot_wo / ((n_dot_wo * (1.0 - k)) + k + eps) # (N,L)
    G = (G_wi * G_wo).clamp(max=1.0)     # (N,L)

    # F : Fresnel equation
    F0 = (1.0 - met) * 0.04 + met * alb # (N,1,3)
    fresnel = F0 + (1.0 - F0) * ((1.0 - wo_dot_h.unsqueeze(-1)) ** 5) # (N,L,3)

    # Specular f
    spec_denom = (4.0 * n_dot_wo * n_dot_wi).clamp(min=eps)  # (N,L)
    f_s = (D * G).unsqueeze(-1) * fresnel / spec_denom.unsqueeze(-1)  # (N,L,3)
    f_s = f_s.clamp(max=10.0)

    # Diffuse f
    f_d = ((1.0 - fresnel)*(1.0 - met)) * alb / torch.pi  # (N,1,3)

    return f_s + f_d    # (N,L,3)


# Rendering step
def render_nerfactor(view_dirs, normal_pred,
                     albedo, metallic, roughness, env_light,
                     l_xyz, areas, visibility=None):
    """
    Evaluate rendering equation as given in NeRFactor paper (Eq. 14)

    Args:
        view_dirs:      (N,3) viewing directions toward camera
        normal_pred:    (N,3) predicted normals from Normal MLP
        albedo:         (N,3) predicted diffuse albedo
        metallic:       (N,1) predicted metallic
        roughness:      (N,1) predicted roughness
        env_light:      Environment Light Map, L
        l_xyz:          (L,3) light direction vectors
        areas:          (L,1) solid angles per light pixel
        visibility:     (N,L) visibility of each pairs of point and light direction 

    Returns:
        rgb: (N,3) outgoing radiance / pixel colour in [0,1]

    ** if visibility=None, them Visibility is assumed to be == 1 (All surface points are fully lit and visible.)
    """

    # Environment map
    light_map = env_light()     # shape (1,3,H,W)
    light_vals = light_map.squeeze(0).permute(1,2,0).reshape(-1,3)  # shape (L,3)
    L_i = light_vals.unsqueeze(0)   # (1,L,3)

    # Prepare shapes
    dw = areas.unsqueeze(0) # (1,L,1)
    pred_norm = torch.sqrt(torch.sum(normal_pred**2, dim=-1, keepdim=True) + 1e-8)
    normals = (normal_pred / pred_norm).unsqueeze(1)        # (N,1,3)
    w_o = F.normalize(view_dirs, dim=-1).unsqueeze(1)       # (N,1,3)
    w_i = l_xyz.unsqueeze(0)    # (1,L,3)

    # Incident cosine term, light is weaker at grazing angle (almost parallel to surface)
    cos_i = (normals * w_i).sum(-1).clamp(min=0.0)  # shape (N,L)
    cos = cos_i.unsqueeze(-1)   # (N,L,1)

    # Cook-Torrance BRDF
    brdf = cooktorrance_brdf(albedo, metallic, roughness, normals, w_i, w_o)    # (N,L,3)

    # Visibility 
    if visibility is not None:
        V = visibility.unsqueeze(-1)    # (N,L,1)
        rgb = (brdf * L_i * cos * dw * V).sum(dim=1)   # shape (N,3)

    else:
        # Rendering integral
        rgb = (brdf * L_i * cos * dw).sum(dim=1)   # shape (N,3)

    return rgb.clamp(0.0,1.0)

ACC_THRESH = 0.1
GT_ALPHA_THRESH = 0.5

# Final/Total loss function for NeRFactor
def nerfactor_loss(pred_rgb, gt_rgb, alphas,
                   pred_normal, gt_normal, jitter_normal,
                   pred_albedo, jitter_albedo,
                   pred_metal, jitter_metal,
                   pred_rough, jitter_rough,
                   env_light, 
                   # Loss weights hyperparameters (λ1, ..., λ7 in NeRFactor paper)
                   w_normal_loss = 0.1, # orig 0.1
                   w_normal_smooth = 0.05,   # orig 0.05
                   w_albedo_smooth = 0.05,   # orig 0.05
                   w_material_smooth = 0.005,    # orig 0.005
                   w_lighting = 5e-4,       # orig 5e-6
                   w_white_balance = 5e-4   # orig 5e-6
                   ):

    """
    Compute all losses in NeRFactor.

    Full loss: ℓ_recon + ℓ_n + ℓ_a + ℓ_mat + ℓ_i

    Returns:
        total_loss: Scalar tensor of final loss
        loss_dict:  Dict of all loss components
    """

    ## Reconstruction loss
    l_recon = F.mse_loss(pred_rgb, gt_rgb)

    ## Normal MLP loss
    l_normal_data = ((pred_normal - gt_normal).pow(2).sum(dim=-1).mean()) * (w_normal_loss / 3.0)
    l_normal_smooth = F.l1_loss(pred_normal, jitter_normal) * (w_normal_smooth / 3.0)

    ## Albedo MLP loss
    l_albedo = F.l1_loss(pred_albedo, jitter_albedo) * (w_albedo_smooth / 3.0)

    ## Material loss
    l_metal = F.l1_loss(pred_metal, jitter_metal) * w_material_smooth 
    l_rough = F.l1_loss(pred_rough, jitter_rough) * w_material_smooth 
    l_material = (l_metal + l_rough) 

    ## Lighting penalties
    l_white_balance = env_light.white_balance_penalty() * w_white_balance
    l_lighting = (env_light.gradient_penalty_spatial() * w_lighting) + (env_light.gradient_penalty_chromatic() * w_lighting) + l_white_balance

    ## Total loss
    l_total = l_recon + l_normal_data + l_normal_smooth + l_albedo + l_material + l_lighting

    ## Loss dict
    loss_dict = {
        "recon" : l_recon.item(),
        "normal" : (l_normal_data + l_normal_smooth).item(),
        "albedo" : l_albedo.item(),
        "roughness" : l_metal.item(),
        "metallic" : l_rough.item(),
        "lighting" : l_lighting.item(),
        "white-balance" : l_white_balance.item(),
        "total" : l_total.item()
    }

    return l_total, loss_dict


def evaluate_nerfactor(ingp_model, nerfactor_mlp, env_light, 
                       eval_images, eval_alphas, eval_poses, init_raydir, init_rayorig,
                       N, t_bin_leftbound, t_gap, chunk_size, device, l_xyz, areas,
                       xyz_jitter_std, max_images=None, lambda_val=None, scene_bounds=SCENE_BOUNDS, use_vanilla=False, mitsuba_scene=None):
    """
    Evaluate model on a set of images and poses based on NeRFactor model.
    Returns average loss dictionary and PSNR across all evaluated images.
    """

    nerfactor_mlp.eval()
    env_light.eval()

    losses_dict = {
        "recon" : 0.0,
        "normal" : 0.0,
        "albedo" : 0.0,
        "roughness" : 0.0,
        "metallic" : 0.0,
        "lighting" : 0.0,
        "white-balance" : 0.0,
        "total" : 0.0
    } 
    psnrs  = []

    n_eval = len(eval_images) if max_images is None else min(max_images, len(eval_images))
    random_pick = np.random.randint(0, high=n_eval)
    C_r_return = None
    alb_return = None
    met_return = None
    rou_return = None
    normal_return = None
    normal_gt_return = None
    vis_return = None

    for i in range(n_eval):
        # Get pose for this image
        c2w = torch.FloatTensor(eval_poses[i]).to(device)
        rot = c2w[:3, :3]
        orig = c2w[:3, -1]

        # Build rays
        raydir  = torch.einsum("ij,hwj->hwi", rot, init_raydir)
        rayorig = orig.expand(raydir.shape)
        target = torch.FloatTensor(eval_images[i]).to(device)
        target_alpha = torch.FloatTensor(eval_alphas[i]).to(device) 

        H, W, _ = raydir.shape
        total_pixels = H * W

        with torch.no_grad():
            # Get geometry from frozen INGP NeRF model
            r_s, t_s = get_coarse_samples(raydir, N, t_bin_leftbound, t_gap, rayorig)
            _, w_s = render_volume(r_s, raydir, chunk_size, ingp_model, t_s, lambda_val=lambda_val)

            # Get surface points & view direction
            w_s_flat = w_s.reshape(-1,N)
            acc_flat = w_s_flat[:,:-1].sum(dim=-1)
            sp = compute_surface_pts(raydir.reshape(-1,3), rayorig.reshape(-1,3), t_s.reshape(-1,N)[:,:-1], w_s_flat[:,:-1])
            vd = -raydir.reshape(-1,3)

            gt_alpha_flat = target_alpha.reshape(-1)
            gt_mask = gt_alpha_flat > GT_ALPHA_THRESH

            acc_composite = gt_alpha_flat.float()   # ground truth mask for compositing and loss

            sp = sp[gt_mask]
            vd = vd[gt_mask]
            acc_flat_ = acc_flat[gt_mask]
            target = target.reshape(-1,3)[gt_mask]
            acc_composite = acc_composite[gt_mask]
            # oob_mask = ~in_scene_bounds(sp, bounds=scene_bounds)
            # acc_flat = torch.where(oob_mask, torch.zeros_like(acc_flat), acc_flat)

        # Evaluate MLPs in chunks
        all_C_r = []
        all_normal_pred = []
        all_normal_analytic = []
        all_albedo_pred = []
        all_met_pred = []
        all_rou_pred = []
        all_vis = []

        all_normal_jit = []
        all_albedo_jit = []
        all_met_jit = []
        all_rou_jit = []

        for chunk in range(0, sp.shape[0], chunk_size):
            sp_chunk = sp[chunk : chunk + chunk_size]
            vd_chunk = vd[chunk : chunk + chunk_size]
            acc_chunk = acc_flat_[chunk : chunk + chunk_size]
            acc_comp_chunk = acc_composite[chunk : chunk + chunk_size]

            # Jittered points
            sp_chunk_jit = sp_chunk + torch.randn_like(sp_chunk) * xyz_jitter_std

            # Ground truth normals
            mask = acc_chunk > ACC_THRESH
            normal_analytic_chunk = torch.zeros(len(acc_chunk), 3, device=device)
            if mask.any():
                if use_vanilla:
                    normal_analytic_chunk[mask]  = compute_analytic_normals_vanilla(ingp_model, sp_chunk[mask], scene_bounds=scene_bounds)
                else:
                    normal_analytic_chunk[mask]  = compute_analytic_normals(ingp_model, sp_chunk[mask], scene_bounds=scene_bounds)

            with torch.no_grad():
                # Forward pass
                np_chunk, ap_chunk, mp_chunk, rp_chunk = nerfactor_mlp(sp_chunk)
                nj_chunk, aj_chunk, mj_chunk, rj_chunk = nerfactor_mlp(sp_chunk_jit)


            # Visibility
            L = l_xyz.shape[0]
            vis_chunk = torch.ones(len(acc_chunk), L, device=device)   # initialise as fully visible
            if mitsuba_scene is not None and mask.any():
                vis = compute_visibility(mitsuba_scene, sp_chunk[mask].detach().cpu().numpy(), 
                                         np_chunk[mask].detach().cpu().numpy(),
                                         l_xyz.detach().cpu().numpy())
                vis_chunk[mask] = torch.tensor(vis, dtype=torch.float32, device=device) # (N_valid, L)
            vis_mean = vis_chunk.mean(dim=1)    # (chunk,)


            with torch.no_grad():
                C_r_chunk = render_nerfactor(vd_chunk, np_chunk, ap_chunk, 
                                        mp_chunk, rp_chunk, env_light,
                                        l_xyz, areas, visibility=vis_chunk)

                acc_chunk_ = acc_comp_chunk.unsqueeze(-1).clamp(0.0, 1.0)
                
                C_r_chunk = C_r_chunk * acc_chunk_ + (1.0 - acc_chunk_) 


            all_C_r.append(C_r_chunk)
            all_normal_pred.append(np_chunk)
            all_normal_analytic.append(normal_analytic_chunk)
            all_albedo_pred.append(ap_chunk)
            all_met_pred.append(mp_chunk)
            all_rou_pred.append(rp_chunk)
            all_vis.append(vis_mean)
            all_normal_jit.append(nj_chunk)
            all_albedo_jit.append(aj_chunk)
            all_met_jit.append(mj_chunk)
            all_rou_jit.append(rj_chunk)

        # Concatenate all chunks
        C_r_full = torch.cat(all_C_r, dim=0)
        normal_pred_full = torch.cat(all_normal_pred, dim=0)
        normal_analytic_full = torch.cat(all_normal_analytic, dim=0) 
        albedo_pred_full = torch.cat(all_albedo_pred, dim=0)
        met_pred_full = torch.cat(all_met_pred, dim=0)
        rou_pred_full = torch.cat(all_rou_pred, dim=0)
        vis_full = torch.cat(all_vis, dim=0)
        normal_jit_full = torch.cat(all_normal_jit, dim=0)
        albedo_jit_full = torch.cat(all_albedo_jit, dim=0)
        met_jit_full = torch.cat(all_met_jit, dim=0)
        rou_jit_full = torch.cat(all_rou_jit, dim=0)

        # Compute total loss
        target_flat = target.reshape(-1,3)
        gt_alpha_flat = gt_alpha_flat[gt_mask]
        l_total, loss_dict = nerfactor_loss(
            C_r_full, target_flat, gt_alpha_flat,
            normal_pred_full, normal_analytic_full, normal_jit_full,
            albedo_pred_full, albedo_jit_full,
            met_pred_full, met_jit_full,
            rou_pred_full, rou_jit_full,
            env_light
        )
        
        # PSNR based only on reconstruction loss
        psnr   = 10.0 * torch.log10(torch.tensor(1.0 / loss_dict["recon"]))

        for k in losses_dict.keys():
            losses_dict[k] += loss_dict[k]

        psnrs.append(psnr.item())

        if i == random_pick:
            # Create full-size tensors initialized with default values
            C_r_full_img = torch.zeros(total_pixels, 3, device=device)
            albedo_full_img = torch.zeros(total_pixels, 3, device=device)
            met_full_img = torch.zeros(total_pixels, 1, device=device)
            rou_full_img = torch.zeros(total_pixels, 1, device=device)
            normal_full_img = torch.zeros(total_pixels, 3, device=device)
            normal_gt_full_img = torch.zeros(total_pixels, 3, device=device)
            vis_full_img = torch.zeros(total_pixels, 1, device=device)

            # Place filtered results into full image
            C_r_full_img[gt_mask] = C_r_full
            albedo_full_img[gt_mask] = albedo_pred_full
            met_full_img[gt_mask] = met_pred_full
            rou_full_img[gt_mask] = rou_pred_full
            normal_full_img[gt_mask] = normal_pred_full
            normal_gt_full_img[gt_mask] = normal_analytic_full
            vis_full_img[gt_mask] = vis_full.unsqueeze(-1)

            # Reshape for visualization
            gt_alpha_img = target_alpha.reshape(H, W, 1)
            comp_mask = gt_alpha_img > GT_ALPHA_THRESH
            comp_mask = comp_mask.float().cpu()

            # comp_mask for compositing
            C_r_return = C_r_full_img.reshape(H, W, 3).cpu() * comp_mask + (1.0 - comp_mask)
            alb_return = albedo_full_img.reshape(H, W, 3).cpu() * comp_mask + (1.0 - comp_mask)
            
            # For metallic and roughness, composite onto black (not white)
            met_return = met_full_img.reshape(H, W, 1).repeat(1, 1, 3).cpu() * comp_mask
            rou_return = rou_full_img.reshape(H, W, 1).repeat(1, 1, 3).cpu() * comp_mask
            
            # For normals, composite with white background
            normal_norm = F.normalize(normal_full_img, dim=-1)
            normal_return = ((normal_norm.reshape(H, W, 3) + 1.0) / 2.0).cpu() * comp_mask + (1.0 - comp_mask)
            
            normal_gt_norm = F.normalize(normal_gt_full_img, dim=-1)
            normal_gt_return = ((normal_gt_norm.reshape(H, W, 3) + 1.0) / 2.0).cpu() * comp_mask + (1.0 - comp_mask)

            # Visibility, composite with white background, grayscale
            vis_return = vis_full_img.reshape(H, W, 1).repeat(1, 1, 3).cpu() * comp_mask + (1.0 - comp_mask)

    # Average the metrics
    for k in losses_dict.keys():
        losses_dict[k] /= n_eval

    return losses_dict, np.mean(psnrs), C_r_return, alb_return, met_return, rou_return, normal_return, normal_gt_return, vis_return, random_pick


################################################################################################################################################################
# TSDF Fusion Helper Functions
################################################################################################################################################################

def pose_nerf2opencv(pose):
    """
    Converts NeRF convention pose to that of OpenCV.

    NeRF:   camera looks down -z, y is up   (4,4)
    OpenCV: camera looks down +z, y is down (4,4)
    """

    flip = np.diag([1, -1, -1, 1]).astype(np.float32)
    return pose @ flip


def get_camera_intrinsics(focal, height, width):
    """
    Create camera 3x3 intrinsics matrix.
    """

    return np.array([
        [focal, 0, (width - 1.0) / 2.0],
        [0, focal, (height - 1.0) / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)


def get_volume_bounds(poses, scene_bounds, voxel_size):
    """
    Create volume bounds from defined scene bounds.
    """

    min_bound = np.array(scene_bounds[0])
    max_bound = np.array(scene_bounds[1])

    vol_bounds = np.stack([min_bound, max_bound], axis=1)   # shape (3,2)

    # ensure bounds are multiples of voxel size
    dims = np.ceil((vol_bounds[:,1] - vol_bounds[:,0]) / voxel_size)
    vol_bounds[:,1] = vol_bounds[:,0] + dims * voxel_size

    return vol_bounds


def render_full(model, pose, init_raydir, init_rayorig,
                N_total, t_bin_leftbound, t_gap, chunk_size,
                height, width, device, lambda_val=None, row_chunk=32, gt_alpha=None):
    """
    Render full image of depth map and colour image.

    Returns:
        (H,W) depth map
        (H,W,3) RGB image
        (H,W) boolean mask of valid pixels
    """

    model.eval()

    depth = torch.zeros(height, width, device=device)
    colours = torch.zeros(height, width, 3, device=device)
    acc = torch.zeros(height, width, device=device)

    with torch.no_grad():
        # Get pose
        pose_t = torch.Tensor(pose).to(device)
        rot = pose_t[:3,:3]
        orig = pose_t[:3,3]

        # Setup rays
        raydir = torch.einsum("ij,hwj->hwi", rot, init_raydir)
        rayorig = orig.expand(raydir.shape)

        for r0 in range(0, height, row_chunk):
            r1 = min(r0 + row_chunk, height)
            raydir_block = raydir[r0:r1]
            rayorig_block = rayorig[r0:r1]

            r_sampled, t_sampled = get_coarse_samples(raydir_block, N_total, t_bin_leftbound, t_gap, rayorig_block)
            colours_block, weights_block = render_volume(r_sampled, raydir_block, chunk_size, model, t_sampled, lambda_val=lambda_val)
            depth[r0:r1] = (weights_block * t_sampled).sum(-1)
            acc[r0:r1] = weights_block.sum(-1)
            colours[r0:r1] = colours_block

    valid_mask = acc > GT_ALPHA_THRESH    

    if gt_alpha is not None:
        gt_mask_t = torch.tensor(gt_alpha, device=device) > GT_ALPHA_THRESH
        valid_mask = valid_mask & gt_mask_t

    depth[~valid_mask] = 0.0
    return depth.cpu().numpy().astype(np.float64), (colours.clamp(0,1).cpu().numpy()*255).astype(np.uint8), valid_mask.cpu().numpy()


def fuse_into_mesh(ingp_model, poses, idx_train, init_raydir, init_rayorig,
                   height, width, focal, device,
                   N_total, t_bin_leftbound, t_gap, chunk_size,
                   scene_bounds=SCENE_BOUNDS, voxel_size=0.02, use_gpu=False, max_views=None, lambda_val=None, gt_alpha=None, output_path="mesh.ply"):
    """
    Fuse multiple views of NeRF model to form a TSDF volume and extract a mesh from it.
    """
    import fusion

    print("=" * 80)
    print("Starting TSDF Fusion ...")
    print("=" * 80)

    # Create camera intrinsics
    cam_intrinsics = get_camera_intrinsics(focal, height, width)
    print(f"Camera intrinsics\n {cam_intrinsics}")

    # Views to be used
    if max_views is not None and len(idx_train) > max_views:
        # evenly spaced indices
        step = len(idx_train) // max_views
        selected_indices = idx_train[::step][:max_views]
        print(f"Using {len(selected_indices)} of {len(idx_train)} training views")
    else:
        selected_indices = idx_train
        print(f"Using all {len(selected_indices)} training views")

    # Volume bounds
    vol_bounds = get_volume_bounds(poses[selected_indices], scene_bounds, voxel_size)
    print(f"Volume bounds:\n{vol_bounds}")
    print(f"Voxel size: {voxel_size} m")

    # TSDF Volume
    tsdf_vol = fusion.TSDFVolume(vol_bounds, voxel_size, use_gpu=use_gpu)

    # Fuse each view
    print("\nFusing views ...")
    ingp_model.eval()
    n_views = len(selected_indices)
    t_start = time.time()

    for i, idx in enumerate(selected_indices):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Fusing view {i+1}/{n_views} (pose {idx})")
        
        pose = poses[idx]
        depth, colour, valid_mask = render_full(ingp_model, pose, init_raydir, init_rayorig,
                                                N_total, t_bin_leftbound, t_gap, chunk_size,
                                                height, width, device, lambda_val=lambda_val, gt_alpha=gt_alpha[idx])
        pose_cv = pose_nerf2opencv(pose)    # OpenCV pose convention

        # integrate into TSDF volume
        try:
            tsdf_vol.integrate(colour, depth, cam_intrinsics, pose_cv, obs_weight=1.0)
        except Exception as e:
            print(f"    WARNING Failed to integrate view {idx}: {e}")
            continue

    elapsed_time = time.time() - t_start
    print(f"\nFusion complete in {elapsed_time:.2f} seconds ({n_views/elapsed_time:.2f} FPS)")

    # Mesh extraction
    print(f"\nExtracting mesh from TSDF volume...")
    verts, faces, norms, colours = tsdf_vol.get_mesh()
    colours_norm = colours.astype(np.float32) / 255.0
    fusion.meshwrite(output_path, verts, faces, norms, colours_norm)
    print(f"Mesh saved to {output_path} successfully.")
    print(f"  Vertices: {len(verts):,}")
    print(f"  Faces: {len(faces):,}")

    return tsdf_vol


def clean_mesh(mesh_path):
    """
    Clean the generated mesh from disconnected components and etc.
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # clean degenerate geometry
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    # remove disconnected components, save only largest connected component
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) > 1:
        # Find the index of the cluster with the most triangles
        largest_cluster_idx = cluster_n_triangles.argmax()
        
        # Create a boolean mask of triangles that do NOT belong to the main cluster
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        
        # Remove the dust and clean up the unreferenced vertices
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        
        print(f"Removed {len(cluster_n_triangles) - 1} disconnected dust components")

    o3d.io.write_triangle_mesh(mesh_path, mesh)


################################################################################################################################################################
# Mitsuba / Visibility Helper Functions
################################################################################################################################################################

def load_mitsuba_scene(mesh_path):
    """
    Loads mesh as a Mitsuba scene.
    """

    return mi.load_dict({
        'type' : 'scene',
        'shape' : {
            'type' : 'ply',
            'filename' : mesh_path,
            'face_normals' : False
        }
    })


def compute_visibility(scene, surface_pts, normals, l_xyz, eps=0.05, radius=0.05, chunk_size=1<<20):
    """
    Computes visibility of each surface point with each possible light direction.

    Returns:
        (N,L) Boolean array of visibility, where point is visible == True.
    """
    # normalize normals
    normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    invalid = (normal_norm.squeeze(-1) < 1e-5)
    normals = normals / (normal_norm + 1e-8)

    # fallback for degenerate/zero normals
    if invalid.any():
        normals[invalid] = np.array([0.0, 0.0, 1.0], dtype=normals.dtype)

    N, L = surface_pts.shape[0], l_xyz.shape[0] # (N,3), (L,3)

    # setup 
    pts_repeat = np.repeat(surface_pts, L, axis=0)  # (N*L, 3)
    normal_repeat = np.repeat(normals, L, axis=0)   # (N*L, 3)
    dir_repeat = np.tile(l_xyz, (N,1))              # (N*L, 3)
    # offset ray origin along normal to prevent self-intersection (shadow acne) + offset along ray direction for larger acceptable window of visibility
    origins = pts_repeat + (normal_repeat * eps) + (dir_repeat * radius)      

    vis_flat = np.zeros(N*L, dtype=bool)

    for start in range(0, N*L, chunk_size):
        end = min(start + chunk_size, N*L)
        rays = mi.Ray3f(
            o=mi.Point3f(origins[start:end, 0], origins[start:end, 1], origins[start:end, 2]),
            d=mi.Vector3f(dir_repeat[start:end, 0], dir_repeat[start:end, 1], dir_repeat[start:end, 2])
        )
        rays.maxt = dr.inf  # max position on ray segment set to infinity

        occluded = np.array(scene.ray_test(rays))   # returns True if ray hits something ==> not visible
        vis_flat[start:end] = ~occluded             # visibility set to opposite of occluded 

    vis = vis_flat.reshape(N,L)
    vis[invalid] = True
    return vis


################################################################################################################################################################
# ScanNet++ Mesh Extraction from INGP NeRF Helpers
################################################################################################################################################################

def get_camera_intrinsics_pinhole(fx, fy, cx, cy):
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def render_full_box(model, pose, init_raydir, N_total, chunk_size,
                     height, width, device, lambda_val=None, row_chunk=32,
                     bound_min=-1.0, bound_max=1.0, weight_thresh=0.5):
    """
    Render full depth/colour/valid-mask for a ScanNet++-trained INGP model.
    Uses ray-box sampling (get_coarse_samples_box).
    No ground-truth alpha exists for real scenes, so validity comes from the
    NeRF's own accumulated ray weight.
    """
    model.eval()
    depth = torch.zeros(height, width, device=device)
    colours = torch.zeros(height, width, 3, device=device)
    acc = torch.zeros(height, width, device=device)

    with torch.no_grad():
        pose_t = torch.Tensor(pose).to(device)
        rot = pose_t[:3, :3]
        orig = pose_t[:3, 3]

        raydir = torch.einsum("ij,hwj->hwi", rot, init_raydir)
        rayorig = orig.expand(raydir.shape)

        for r0 in range(0, height, row_chunk):
            r1 = min(r0 + row_chunk, height)
            raydir_block = raydir[r0:r1]
            rayorig_block = rayorig[r0:r1]

            r_sampled, t_sampled = get_coarse_samples_box(
                raydir_block, rayorig_block, N_total,
                bound_min=bound_min, bound_max=bound_max
            )
            colours_block, weights_block = render_volume(
                r_sampled, raydir_block, chunk_size, model, t_sampled, lambda_val=lambda_val
            )

            depth[r0:r1] = (weights_block * t_sampled).sum(-1)
            acc[r0:r1] = weights_block.sum(-1)
            colours[r0:r1] = colours_block

    valid_mask = acc > weight_thresh
    depth[~valid_mask] = 0.0

    return (depth.cpu().numpy().astype(np.float64),
            (colours.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            valid_mask.cpu().numpy())


def fuse_into_mesh_scannetpp(ingp_model, poses, fl_x, fl_y, cx, cy, idx_train,
                              init_raydir, height, width, device,
                              N_total, chunk_size,
                              norm_center, norm_scale,
                              voxel_size_m=0.04, weight_thresh=0.5,
                              max_views=None, lambda_val=None,
                              output_path="mesh.ply"):
    import open3d as o3d

    print("=" * 80)
    print("Starting TSDF Fusion (ScanNet++, normalized coordinate space) ...")
    print("=" * 80)

    if max_views is not None and max_views > 0 and len(idx_train) > max_views:
        step = len(idx_train) // max_views
        selected_indices = idx_train[::step][:max_views]
        print(f"Using {len(selected_indices)} of {len(idx_train)} training views")
    else:
        selected_indices = idx_train
        print(f"Using all {len(selected_indices)} training views")

    # Scene lives in roughly [-0.91, 0.91]^3 after compute_scene_normalization's
    # 1.1x padding -- convert your desired METRIC voxel size into this space.
    voxel_size_norm = voxel_size_m / norm_scale
    sdf_trunc_norm = 3.0 * voxel_size_norm
    depth_trunc_norm = 2.5  # comfortably covers the padded unit cube's diagonal

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size_norm,
        sdf_trunc=sdf_trunc_norm,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    print(f"Voxel size: {voxel_size_m} m -> {voxel_size_norm:.5f} normalized units")

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fl_x, fl_y, cx, cy)

    ingp_model.eval()
    n_views = len(selected_indices)
    t_start = time.time()

    for i, idx in enumerate(selected_indices):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Fusing view {i+1}/{n_views} (pose {idx})")

        pose = poses[idx]
        depth_norm, colour, valid_mask = render_full_box(
            ingp_model, pose, init_raydir, N_total, chunk_size,
            height, width, device, lambda_val=lambda_val, weight_thresh=weight_thresh
        )

        pose_cv_c2w = pose_nerf2opencv(pose)          # axis-convention fix only
        extrinsic_w2c = np.linalg.inv(pose_cv_c2w)    # Open3D wants world->camera

        colour_o3d = o3d.geometry.Image(np.ascontiguousarray(colour))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_norm).astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colour_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc_norm,
            convert_rgb_to_intensity=False
        )

        try:
            volume.integrate(rgbd, o3d_intrinsic, extrinsic_w2c.astype(np.float64))
        except Exception as e:
            print(f"    WARNING Failed to integrate view {idx}: {e}")
            continue

    elapsed_time = time.time() - t_start
    print(f"\nFusion complete in {elapsed_time:.2f}s ({n_views/elapsed_time:.2f} FPS)")

    print("\nExtracting mesh ...")
    mesh = volume.extract_triangle_mesh()

    # Undo the scene normalization so the saved mesh is in real-world metres
    # verts_world = np.asarray(mesh.vertices) * norm_scale + norm_center
    # mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    # mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Mesh saved to {output_path} (world-scale metres)")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.triangles):,}")

    return volume


## Vanilla NeRF for ScanNet++

def render_full_box_vanilla(model_c, model_f, pose, init_raydir, N_c, N_f, chunk_size,
                             height, width, device, row_chunk=32,
                             bound_min=-1.0, bound_max=1.0, weight_thresh=0.5):
    """
    Render full depth/colour/valid-mask for a ScanNet++-trained Vanilla NeRF
    (separate coarse + fine MLPs). Ray-box sampling gives the coarse bounds,
    then get_fine_samples does the usual importance-sampling refinement.
    """
    model_c.eval()
    model_f.eval()

    depth = torch.zeros(height, width, device=device)
    colours = torch.zeros(height, width, 3, device=device)
    acc = torch.zeros(height, width, device=device)

    with torch.no_grad():
        pose_t = torch.Tensor(pose).to(device)
        rot = pose_t[:3, :3]
        orig = pose_t[:3, 3]

        raydir = torch.einsum("ij,hwj->hwi", rot, init_raydir)
        rayorig = orig.expand(raydir.shape)

        for r0 in range(0, height, row_chunk):
            r1 = min(r0 + row_chunk, height)
            raydir_block = raydir[r0:r1]
            rayorig_block = rayorig[r0:r1]

            # --- coarse pass: sample along the ray/box intersection ---
            r_c, t_c = get_coarse_samples_box(
                raydir_block, rayorig_block, N_c,
                bound_min=bound_min, bound_max=bound_max
            )
            _, weights_c = render_volume(r_c, raydir_block, chunk_size, model_c, t_c)

            # far bound of this block, needed by get_fine_samples' last-bin cutoff
            flat_o = rayorig_block.reshape(-1, 3)
            flat_d = raydir_block.reshape(-1, 3)
            _, far = neuh.get_near_far_bounds_box(flat_o, flat_d, bound_min, bound_max)
            far_valid = far[torch.isfinite(far)]
            t_f_bound = float(far_valid.max().item()) if far_valid.numel() > 0 else 2.5

            # --- fine pass: importance sample + combine with coarse t's ---
            r_f, t_f = get_fine_samples(weights_c, N_f, t_c, t_f_bound, rayorig_block, raydir_block)
            colours_block, weights_block = render_volume(r_f, raydir_block, chunk_size, model_f, t_f)

            depth[r0:r1] = (weights_block * t_f).sum(-1)
            acc[r0:r1] = weights_block.sum(-1)
            colours[r0:r1] = colours_block

    valid_mask = acc > weight_thresh
    depth[~valid_mask] = 0.0

    return (depth.cpu().numpy().astype(np.float64),
            (colours.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            valid_mask.cpu().numpy())


def fuse_into_mesh_scannetpp_vanilla(model_c, model_f, poses, fl_x, fl_y, cx, cy, idx_train,
                                      init_raydir, height, width, device,
                                      N_c, N_f, chunk_size,
                                      norm_center, norm_scale,
                                      voxel_size_m=0.04, weight_thresh=0.5,
                                      max_views=None, output_path="mesh.ply"):
    import open3d as o3d

    print("=" * 80)
    print("Starting TSDF Fusion (ScanNet++, Vanilla NeRF, normalized coordinate space) ...")
    print("=" * 80)

    if max_views is not None and max_views > 0 and len(idx_train) > max_views:
        step = len(idx_train) // max_views
        selected_indices = idx_train[::step][:max_views]
        print(f"Using {len(selected_indices)} of {len(idx_train)} training views")
    else:
        selected_indices = idx_train
        print(f"Using all {len(selected_indices)} training views")

    voxel_size_norm = voxel_size_m / norm_scale
    sdf_trunc_norm = 3.0 * voxel_size_norm
    depth_trunc_norm = 2.5

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size_norm,
        sdf_trunc=sdf_trunc_norm,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    print(f"Voxel size: {voxel_size_m} m -> {voxel_size_norm:.5f} normalized units")

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fl_x, fl_y, cx, cy)

    model_c.eval()
    model_f.eval()
    n_views = len(selected_indices)
    t_start = time.time()

    for i, idx in enumerate(selected_indices):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Fusing view {i+1}/{n_views} (pose {idx})")

        pose = poses[idx]
        depth_norm, colour, valid_mask = render_full_box_vanilla(
            model_c, model_f, pose, init_raydir, N_c, N_f, chunk_size,
            height, width, device, weight_thresh=weight_thresh
        )

        pose_cv_c2w = pose_nerf2opencv(pose)
        extrinsic_w2c = np.linalg.inv(pose_cv_c2w)

        colour_o3d = o3d.geometry.Image(np.ascontiguousarray(colour))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_norm).astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colour_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc_norm,
            convert_rgb_to_intensity=False
        )

        try:
            volume.integrate(rgbd, o3d_intrinsic, extrinsic_w2c.astype(np.float64))
        except Exception as e:
            print(f"    WARNING Failed to integrate view {idx}: {e}")
            continue

    elapsed_time = time.time() - t_start
    print(f"\nFusion complete in {elapsed_time:.2f}s ({n_views/elapsed_time:.2f} FPS)")

    print("\nExtracting mesh ...")
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Mesh saved to {output_path}")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.triangles):,}")

    return volume
