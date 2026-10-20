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
mi.set_variant('cuda_ad_rgb')   # cuda_ad_rgb for CUDA / llvm_ad_rgb for GPU
import drjit as dr
import neus_helper as neuh
import trimesh

ACC_THRESH = 0.1
GT_ALPHA_THRESH = 0.5

################################################################################################################################################################
# Blender Synthetic Loader
################################################################################################################################################################


def load_data(basedir, testskip=1):
    """
    Load and prepare data into desired format for NeRF.

    Args:
        basedir:    Path to scene folder
        testskip:   Use every Nth test/val image
    
    Returns:
        images:         (N, H, W, 3) for N images
        poses:          (N, 4, 4), camera-to-world matrices
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
    
    return imgs, imgs_alpha, poses, [H, W, focal], i_split


################################################################################################################################################################
# Material Prior Loader
################################################################################################################################################################


def load_material_priors(obj, img_ids_split, height, width, alphas=None, alpha_thresh=GT_ALPHA_THRESH, basedir="/home/s64jgan_hpc/IntrinsicImageDiffusion/output/"):
    """
    Loads per-view material priors predicted from Intrinsic Image Diffusion.
    
    Returns: albedo (N,H,W,3), metallic (N,H,W,1) and roughness (N,H,W,1) as float32 arrays
    """

    albedo_list, metallic_list, roughness_list = [], [], []
    
    train_ids, val_ids, test_ids = img_ids_split
    
    for image_id in train_ids:    # training priors
        alb_path = os.path.join(basedir, obj, "albedo", f"{image_id}_albedo.png")
        met_path = os.path.join(basedir, obj, "metal", f"{image_id}_metal.png")
        rou_path = os.path.join(basedir, obj, "roughness", f"{image_id}_roughness.png")

        alb = np.array(Image.open(alb_path).convert("RGB")).astype(np.float32) / 255.0
        met = np.array(Image.open(met_path).convert("L")).astype(np.float32) / 255.0
        rou = np.array(Image.open(rou_path).convert("L")).astype(np.float32) / 255.0

        if met.ndim == 2:
            met = met[..., None]

        if rou.ndim == 2:
            rou = rou[..., None]

        albedo_list.append(alb)
        metallic_list.append(met)
        roughness_list.append(rou)

    for image_id in test_ids:  # test priors
        alb_path = os.path.join(basedir, obj, "albedo", f"{image_id}_albedo.png")
        met_path = os.path.join(basedir, obj, "metal", f"{image_id}_metal.png")
        rou_path = os.path.join(basedir, obj, "roughness", f"{image_id}_roughness.png")

        alb_test = np.array(Image.open(alb_path).convert("RGB")).astype(np.float32) / 255.0
        met_test = np.array(Image.open(met_path).convert("L")).astype(np.float32) / 255.0
        rou_test = np.array(Image.open(rou_path).convert("L")).astype(np.float32) / 255.0

        if met_test.ndim == 2:
            met_test = met_test[..., None]

        if rou_test.ndim == 2:
            rou_test = rou_test[..., None]

        albedo_list.append(alb_test)
        metallic_list.append(met_test)
        roughness_list.append(rou_test)

    albedo = np.stack(albedo_list, axis=0)
    metallic = np.stack(metallic_list, axis=0)
    roughness = np.stack(roughness_list, axis=0)

    if alphas is not None:  # masking background --> for Blender synthetic
        fg = (alphas > alpha_thresh).astype(np.float32)[..., None]
        albedo = albedo * fg
        metallic = metallic * fg
        roughness = roughness * fg

    return albedo, metallic, roughness


################################################################################################################################################################
# NeuSFactor Helpers
################################################################################################################################################################


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
        preds = model(r_chunk, raydirs_chunk, lambda_val=lambda_val)
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


def run_ingpnerf_once(raydirs, N, t_bin_leftbound, t_gap, rayorig, chunk_size, model, lambda_val=None):
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
    t_gap_total = (t_bin_leftbound[-1] + t_gap - t_bin_leftbound[0]) / N_total

    t_bin_total = (t_bin_leftbound[0].to(raydirs.device) + torch.arange(N_total, device = raydirs.device) * t_gap_total).to(raydirs.device)

    r_sampled, t_sampled = get_coarse_samples(raydirs, N_total, t_bin_total, t_gap_total, rayorig)
    C_r, weights = render_volume(r_sampled, raydirs, chunk_size, model, t_sampled, lambda_val=lambda_val)

    return C_r, weights, t_sampled


def evaluate_ingp(model, eval_images, eval_poses, init_raydir,
             N, t_bin_leftbound, t_gap, chunk_size, device, loss_fn,
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
                raydir, N, t_bin_leftbound, t_gap, rayorig,
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


# SCENE_BOUNDS = ([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])    # for Blender synthetic
SCENE_BOUNDS = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])    # for ScanNet++

SDF_CUBE_BOUND = 1.0    # Bounds of SDF Network's coordinate cube

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


# Extract surface points, x_surf
# Location at which ray is expected to terminate based on sigma
def compute_surface_pts_and_normals(renderer, ray_dirs, ray_origs, near, far, scene_scale, appearance_embedding=None):
    """
    Compute expected surface points and normals from frozen NeuS SDF.
    """
    renderer_out = renderer.render(ray_origs, ray_dirs, near, far,
                                   background_rgb=torch.ones(3, device=ray_origs.device),
                                   cos_anneal_ratio=1.0, appearance_code=appearance_embedding)
    weights = renderer_out["weights"]
    mid_z_vals = renderer_out["mid_z_vals"]
    gradients = renderer_out["gradients"]
    weight_sum = renderer_out["weight_sum"].squeeze(-1)

    # ## Peak depth
    # peak_idx = weights.argmax(dim=-1) # shape (N,)
    # peak_depth = mid_z_vals.gather(1, peak_idx.unsqueeze(-1)).squeeze(-1)

    ## Expected depth
    expected_depth = renderer_out["depth_fine"].squeeze(-1)

    # x_surf 
    surface_pts = ray_origs + expected_depth.unsqueeze(-1) * ray_dirs
    surface_pts_world = surface_pts * scene_scale   # raw scene coords
    
    # Normals, using expected depth
    normals = (weights.unsqueeze(-1) * gradients).sum(dim=1) # torch.gather(gradients, 1, peak_idx.view(-1,1,1).expand(-1,1,3)).squeeze(1)
    normals = F.normalize(normals, dim=-1)

    # raw sampled points
    sampled_pts = ray_origs[..., None, :] + mid_z_vals[..., :, None] * ray_dirs[..., None, :]

    return surface_pts_world, normals, weight_sum, gradients, sampled_pts, expected_depth


def remesh_scene(renderer, bound_min, bound_max, resolution, scene_scale, mesh_path, light_positions, out_light_path):
    """
    Re-extracts mesh from SDF network and reload Mitsuba scene for light visibility.
    """
    vertices, triangles = renderer.extract_geometry(
        bound_min=bound_min,
        bound_max=bound_max,
        resolution=resolution,
        threshold=0.0  # Surface is where SDF = 0
    )

    mesh = trimesh.Trimesh(vertices=vertices * scene_scale, faces=triangles)
    mesh.export(mesh_path)
    print(f"Remeshed Mesh has bounds: {mesh.bounds}")
    mitsuba_scene = load_mitsuba_scene(mesh_path)

    export_lights_with_mesh(mesh_path, light_positions, out_light_path)

    return vertices, triangles, mitsuba_scene


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
    # k = (alpha / 2.0).clamp(min=eps)         # (N,1), IBL lighting
    k = ((rou.squeeze(-1) + 1.0) ** 2 / 8.0).clamp(min=eps) # direct lighting
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


# # Rendering step
# def render_neusfactor(view_dirs, normal_pred,
#                      albedo, metallic, roughness, env_light,
#                      l_xyz, areas, visibility=None):
#     """
#     Evaluate rendering equation as given in NeRFactor paper (Eq. 14)

#     Args:
#         view_dirs:      (N,3) viewing directions toward camera
#         normal_pred:    (N,3) predicted normals from Normal MLP
#         albedo:         (N,3) predicted diffuse albedo
#         metallic:       (N,1) predicted metallic
#         roughness:      (N,1) predicted roughness
#         env_light:      Environment Light Map, L
#         l_xyz:          (L,3) light direction vectors
#         areas:          (L,1) solid angles per light pixel
#         visibility:     (N,L) visibility of each pairs of point and light direction 

#     Returns:
#         rgb: (N,3) outgoing radiance / pixel colour in [0,1]

#     ** if visibility=None, them Visibility is assumed to be == 1 (All surface points are fully lit and visible.)
#     """

#     # Environment map
#     light_map = env_light()     # shape (1,3,H,W)
#     light_vals = light_map.squeeze(0).permute(1,2,0).reshape(-1,3)  # shape (L,3)
#     L_i = light_vals.unsqueeze(0)   # (1,L,3)

#     # Prepare shapes
#     dw = areas.unsqueeze(0) # (1,L,1)
#     pred_norm = torch.sqrt(torch.sum(normal_pred**2, dim=-1, keepdim=True) + 1e-8)
#     normals = (normal_pred / pred_norm).unsqueeze(1)        # (N,1,3)
#     w_o = F.normalize(view_dirs, dim=-1).unsqueeze(1)       # (N,1,3)
#     w_i = l_xyz.unsqueeze(0)    # (1,L,3)

#     # Incident cosine term, light is weaker at grazing angle (almost parallel to surface)
#     cos_i = (normals * w_i).sum(-1).clamp(min=0.0)  # shape (N,L)
#     cos = cos_i.unsqueeze(-1)   # (N,L,1)

#     # Cook-Torrance BRDF
#     brdf = cooktorrance_brdf(albedo, metallic, roughness, normals, w_i, w_o)    # (N,L,3)

#     # Visibility 
#     if visibility is not None:
#         V = visibility.unsqueeze(-1)    # (N,L,1)
#         rgb = (brdf * L_i * cos * dw * V).sum(dim=1)   # shape (N,3)

#     else:
#         # Rendering integral
#         rgb = (brdf * L_i * cos * dw).sum(dim=1)   # shape (N,3)

#     return rgb.clamp(0.0,1.0)


################################################################################################################################################################
# Rendering Equation with Point Light Sources
################################################################################################################################################################

def render_neusfactor(surface_pts, view_dirs, normal_pred,
                      albedo, metallic, roughness, point_lights, ambient_light,
                      visibility=None, eps=1e-4):
    """
    Evaluate rendering equation.

    Args:
        surface_pts:    (N,3) surface points
        view_dirs:      (N,3) viewing directions toward camera
        normal_pred:    (N,3) predicted normals from Normal MLP
        albedo:         (N,3) predicted diffuse albedo
        metallic:       (N,1) predicted metallic
        roughness:      (N,1) predicted roughness
        point_lights:   (N,3) point light sources
        ambient_light:  (3,)  learnable ambient RGB colour
        visibility:     (N,L) visibility of each pairs of point and light direction 

    Returns:
        rgb: (N,3) outgoing radiance / pixel colour in [0,1]

    ** if visibility=None, them Visibility is assumed to be == 1 (All surface points are fully lit and visible.)
    """                  
    light_pos = point_lights.positions()        # (L,3)
    light_intensity = point_lights.intensities()# (L,3)

    to_light = light_pos.unsqueeze(0) - surface_pts.unsqueeze(1)    # (N,L,3)
    dist2 = (to_light ** 2).sum(-1, keepdim=True)                   # (N,L,1)
    dist = dist2.clamp(min=1e-8).sqrt()
    w_i = to_light / dist                                           # (N,L,3)

    L_emit = point_lights.emission(-w_i)

    pred_norm = torch.sqrt((normal_pred ** 2).sum(-1, keepdim=True) + 1e-8)
    normals = (normal_pred / pred_norm).unsqueeze(1)        # (N,1,3)
    w_o = F.normalize(view_dirs, dim=-1).unsqueeze(1)       # (N,1,3)

    cos_i = (normals * w_i).sum(-1).clamp(min=0.0)  # (N,L)

    brdf = cooktorrance_brdf(albedo, metallic, roughness, normals, w_i, w_o).clamp(max=1.0)   # (N,L,3)
    dist2_falloff = dist2.clamp(min=0.2 ** 2)
    falloff = 1.0 / (dist2_falloff)       # (N,L,1)
    L_i = L_emit * falloff

    contrib = brdf * L_i * cos_i.unsqueeze(-1)
    if visibility is not None:
        contrib = contrib * visibility.unsqueeze(-1)

    rgb_direct = contrib.sum(dim=1)

    ambient_contrib = albedo * ambient_light.get_value().unsqueeze(0)   # (N,3)
    rgb = rgb_direct + ambient_contrib

    return rgb.clamp(min=0.0)


# Final/Total loss function for NeuSFactor
def neusfactor_loss(pred_rgb, gt_rgb,
                    pred_albedo, jitter_albedo,
                    pred_metal, jitter_metal,
                    pred_rough, jitter_rough,
                    point_lights, # env_light, 
                    # Geometry regularizers -------------------------------
                    gradients,                              # raw SDF gradients at ray sample points
                    sampled_pts,
                    surface_pts,
                    aux_tree,
                    aux_verts_t,
                    emitter_positions, emitter_weights,
                    bound,
                    depth_scale, depth_target, peak_depth,
                    pred_mask, gt_mask,                     # alpha
                    albedo_prior, metal_prior, rough_prior, # predicted material priors
                    # Loss weights hyperparameters ------------------------
                    w_albedo_smooth = 0.05,   # orig 0.05
                    w_material_smooth = 0.005,    # orig 0.005
                    w_lighting = 5e-4,       # orig 5e-6
                    w_white_balance = 5e-4,   # orig 5e-6
                    w_light_sparsity = 1e-4,     # for point light sources, orig 1e-4
                    w_light_pos = 1e-3,
                    w_attract = 5e-2,   
                    w_eikonal = 0.1,    # orig 0.1
                    w_mask = 0.1,
                    w_depth = 0.1,
                    w_prior_alb = 0.05,
                    w_prior_met = 0.005,
                    w_prior_rou = 0.005,
                    w_aux_mesh = 0.1   # previously 1.0
                    ):

    """
    Compute all losses in NeRFactor.

    Full loss: ℓ_recon + ℓ_a + ℓ_mat + ℓ_i + ℓ_eik + ℓ_mask + ℓ_prior 

    Returns:
        total_loss: Scalar tensor of final loss
        loss_dict:  Dict of all loss components
    """

    ## Reconstruction loss
    pred_rgb_tonemapped = pred_rgb / (pred_rgb + 1.0)
    l_recon = F.mse_loss(pred_rgb_tonemapped, gt_rgb)

    ## Albedo MLP loss
    l_albedo = F.l1_loss(pred_albedo, jitter_albedo) * (w_albedo_smooth / 3.0)

    ## Material loss
    l_metal = F.l1_loss(pred_metal, jitter_metal) * w_material_smooth 
    l_rough = F.l1_loss(pred_rough, jitter_rough) * w_material_smooth 
    l_material = (l_metal + l_rough) 

    # ## Lighting penalties
    # l_white_balance = env_light.white_balance_penalty() * w_white_balance
    # l_lighting = (env_light.gradient_penalty_spatial() * w_lighting) + (env_light.gradient_penalty_chromatic() * w_lighting) + l_white_balance
    l_light_sparsity = point_lights.sparsity_loss() * w_light_sparsity

    # Light position loss
    l_light_pos = light_pos_loss(point_lights, aux_tree, aux_verts_t) * w_light_pos

    # Light brightness attraction loss
    l_attract = light_attraction_loss(point_lights, emitter_positions, emitter_weights, temperature=5.0) * w_attract

    ## Eikonal loss
    valid = torch.all(sampled_pts.abs() <= bound, dim=-1)
    grad_norm = gradients.norm(dim=-1)
    penalty = (grad_norm - 1.0) ** 2
    if valid.any():
        l_eik = penalty[valid].mean() * w_eikonal
    else:
        l_eik = penalty.mean() * w_eikonal

    ## Mask loss -- anchor zero-level set to object boundary
    l_mask = F.binary_cross_entropy(pred_mask.clamp(1e-3, 1.0 - 1e-3), gt_mask) * w_mask

    ## Depth loss
    valid_depth = (depth_target > 0)
    l_depth = F.l1_loss(peak_depth[valid_depth.flatten()], depth_scale * depth_target.squeeze(-1)[valid_depth.flatten()]) * w_depth

    ## Material priors - albedo, metallic and roughness
    l_prior_alb = F.l1_loss(pred_albedo, albedo_prior) * w_prior_alb
    l_prior_met = F.l1_loss(pred_metal, metal_prior) * w_prior_met
    l_prior_rou = F.l1_loss(pred_rough, rough_prior) * w_prior_rou
    l_prior = l_prior_alb + l_prior_met + l_prior_rou

    ## Auxiliary mesh loss
    l_aux = aux_mesh_loss(surface_pts, aux_tree, aux_verts_t, max_dist=0.1) * w_aux_mesh

    ## Total loss
    l_total = l_recon + l_albedo + l_material + l_light_sparsity + l_light_pos + l_attract + l_eik + l_depth + l_prior + l_aux

    ## Loss dict
    loss_dict = {
        "recon" : l_recon.item(),
        "albedo" : l_albedo.item(),
        "roughness" : l_rough.item(),
        "metallic" : l_metal.item(),
        # "lighting" : l_lighting.item(),
        # "white-balance" : l_white_balance.item(),
        "light_sparsity" : l_light_sparsity.item(),
        "light_pos" : l_light_pos.item(),
        "light_attract" : l_attract.item(),
        "eikonal" : l_eik.item(),
        "mask" : l_mask.item(),
        "depth" : l_depth.item(),
        "prior" : l_prior.item(),
        "aux" : l_aux.item(),
        "total" : l_total.item()
    }

    return l_total, loss_dict


def evaluate_neusfactor(renderer, neusfactor_mlp, point_lights, ambient_light,
                        images_t, alphas_t, poses_t, depths_t, depth_scale, init_raydir,
                        height, width, scene_scale, device, idx_val, aux_tree, aux_verts_t, emitter_positions, emitter_weights,
                        xyz_jitter_std, max_images=3, mitsuba_scene=None, chunk=4096, appearance_embedding=None,
                        alb_prior_t=None, met_prior_t=None, rou_prior_t=None, disable_visib=False,
                        current_w_light_sparsity=None, current_w_light_pos=None, current_w_attract=None,
                        current_w_depth=None, current_w_prior_alb=None, current_w_prior_met=None, current_w_prior_rou=None,
                        current_w_aux_mesh=None):
    """
    Evaluate model on a set of images and poses based on NeuSFactor model.
    Returns average loss dictionary and PSNR across all evaluated images.
    """

    neusfactor_mlp.eval()
    point_lights.eval()
    ambient_light.eval()

    losses_dict = {
        "recon" : 0.0,
        "albedo" : 0.0,
        "roughness" : 0.0,
        "metallic" : 0.0,
        # "lighting" : 0.0,
        # "white-balance" : 0.0,
        "light_sparsity" : 0.0,
        "light_pos" : 0.0,
        "light_attract" : 0.0,
        "eikonal" : 0.0,
        "mask" : 0.0,
        "depth" : 0.0,
        "prior" : 0.0,
        "aux" : 0.0,
        "total" : 0.0
    } 
    psnrs  = []

    n_val = images_t.shape[0]
    val_idxs = np.arange(3) # fixed validation images, for random: np.random.choice(n_val, size=min(max_images, n_val), replace=False)
    random_pick = np.random.randint(0, len(val_idxs))
    return_panels = None

    with torch.enable_grad():   # SDF gradient needs autograd
        for i, idx in enumerate(val_idxs):
            # Get pose for this image
            pose = poses_t[idx].to(device)
            rot, trans = pose[:3,:3], pose[:3,3]

            # Build rays
            raydir_grid = torch.einsum("ij,hwj->hwi", rot, init_raydir)
            rays_d_full = F.normalize(raydir_grid, dim=-1).reshape(-1,3)
            rays_o_full = (trans / scene_scale).expand(rays_d_full.shape)

            target_full = images_t[idx].to(device).reshape(-1,3)
            alpha_full = alphas_t[idx].to(device).reshape(-1)
            gt_mask_full = alpha_full > GT_ALPHA_THRESH
            depth_full = depths_t[idx].to(device).reshape(-1)

            alb_prior_full = alb_prior_t[idx].to(device).reshape(-1,3) if alb_prior_t is not None else None
            met_prior_full = met_prior_t[idx].to(device).reshape(-1,1) if met_prior_t is not None else None
            rou_prior_full = rou_prior_t[idx].to(device).reshape(-1,1) if rou_prior_t is not None else None

            n_rays = rays_d_full.shape[0]
            C_r_img = torch.ones(n_rays, 3, device=device)
            albedo_img = torch.ones(n_rays, 3, device=device)
            metal_img = torch.zeros(n_rays, 1, device=device)
            rough_img = torch.zeros(n_rays, 1, device=device)
            normal_img = torch.ones(n_rays, 3, device=device)
            vis_img = torch.ones(n_rays, 1, device=device)

            batch_losses = {k: [] for k in losses_dict}

            # Appearance embedding
            appearance_code = appearance_embedding(torch.tensor([idx_val[idx]], device=device))

            for start in range(0, n_rays, chunk):
                end = start + chunk
                chunk_mask = gt_mask_full[start:end]
                if chunk_mask.sum() == 0:
                    continue

                ro = rays_o_full[start:end][chunk_mask]
                rd = rays_d_full[start:end][chunk_mask]
                target = target_full[start:end][chunk_mask]
                gt_alpha_chunk = alpha_full[start:end][chunk_mask]
                depth_chunk = depth_full[start:end][chunk_mask]

                alb_prior_chunk = alb_prior_full[start:end][chunk_mask] if alb_prior_full is not None else None
                met_prior_chunk = met_prior_full[start:end][chunk_mask] if met_prior_full is not None else None
                rou_prior_chunk = rou_prior_full[start:end][chunk_mask] if rou_prior_full is not None else None

                near, far = neuh.get_near_far_bounds_box(ro, rd)
                surface_pts, normal_analytic, weight_sum, gradients, sampled_pts, peak_depth = compute_surface_pts_and_normals(renderer, rd, ro, near, far, scene_scale, appearance_embedding=appearance_code)

                albedo_pred, metallic_pred, roughness_pred = neusfactor_mlp(surface_pts)
                view_dirs = F.normalize(-rd, dim=-1)

                light_pos = point_lights.positions()

                vis = torch.ones(surface_pts.shape[0], light_pos.shape[0], device=device)
                if mitsuba_scene is not None:
                    vis_np = compute_visibility(mitsuba_scene, surface_pts.detach().cpu().numpy(),
                                                normal_analytic.detach().cpu().numpy(), light_pos.detach().cpu().numpy(), disable=disable_visib)
                    vis = torch.tensor(vis_np, dtype=torch.float32, device=device)

                C_r = render_neusfactor(surface_pts, view_dirs, normal_analytic, albedo_pred, metallic_pred, roughness_pred, point_lights, ambient_light, visibility=vis)
                
                surface_pts_jit = surface_pts + torch.randn_like(surface_pts) * xyz_jitter_std
                albedo_jit, metallic_jit, roughness_jit = neusfactor_mlp(surface_pts_jit)

                _, loss_dict = neusfactor_loss(C_r, target, albedo_pred, albedo_jit,
                                               metallic_pred, metallic_jit, roughness_pred, roughness_jit, point_lights,
                                               gradients, sampled_pts, surface_pts, aux_tree, aux_verts_t, emitter_positions, emitter_weights, SDF_CUBE_BOUND, 
                                               depth_scale, depth_chunk, peak_depth, weight_sum, gt_alpha_chunk,
                                               alb_prior_chunk, met_prior_chunk, rou_prior_chunk,
                                               w_light_sparsity=current_w_light_sparsity, w_light_pos=current_w_light_pos, w_attract=current_w_attract,
                                               w_depth=current_w_depth, w_prior_alb=current_w_prior_alb, w_prior_met=current_w_prior_met, w_prior_rou=current_w_prior_rou,
                                               w_aux_mesh=current_w_aux_mesh)

                for k, v in loss_dict.items():
                    batch_losses[k].append(v)

                idxs_local = torch.nonzero(chunk_mask, as_tuple=True)[0] + start
                C_r_img[idxs_local] = (C_r / (C_r + 1.0)).detach()
                albedo_img[idxs_local] = albedo_pred.detach()
                metal_img[idxs_local] = metallic_pred.detach()
                rough_img[idxs_local] = roughness_pred.detach()
                normal_img[idxs_local] = normal_analytic.detach()
                vis_img[idxs_local] = vis.mean(dim=1, keepdim=True).detach()

            for k in losses_dict:
                if batch_losses[k]:
                    losses_dict[k] += np.mean(batch_losses[k])

            psnr = 10.0 * np.log10(1.0 / max(np.mean(batch_losses["recon"]), 1e-10))
            psnrs.append(psnr)

            if i == random_pick:
                return_panels = (
                    C_r_img.reshape(height, width, 3).cpu(),
                    albedo_img.reshape(height, width, 3).cpu(),
                    metal_img.reshape(height, width, 1).repeat(1,1,3).cpu(),
                    rough_img.reshape(height, width, 1).repeat(1,1,3).cpu(),
                    normal_img.reshape(height, width, 3).cpu(),
                    vis_img.reshape(height, width, 1).repeat(1,1,3).cpu(),
                    idx,
                )

    for k in losses_dict:
        losses_dict[k] /= len(val_idxs)
    
    neusfactor_mlp.train()
    point_lights.train()
    ambient_light.train()

    return losses_dict, np.mean(psnrs), *return_panels


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


## for environment map
# def compute_visibility(scene, surface_pts, normals, l_xyz, eps=0.05, radius=0.05, chunk_size=1<<20):
#     """   
#     Computes visibility of each surface point with each possible light direction.

#     Returns:
#         (N,L) Boolean array of visibility, where point is visible == True.
#     """
#     # normalize normals
#     normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
#     invalid = (normal_norm.squeeze(-1) < 1e-5)
#     normals = normals / (normal_norm + 1e-8)

#     # fallback for degenerate/zero normals
#     if invalid.any():
#         normals[invalid] = np.array([0.0, 0.0, 1.0], dtype=normals.dtype)

#     N, L = surface_pts.shape[0], l_xyz.shape[0] # (N,3), (L,3)

#     # setup 
#     pts_repeat = np.repeat(surface_pts, L, axis=0)  # (N*L, 3)
#     normal_repeat = np.repeat(normals, L, axis=0)   # (N*L, 3)
#     dir_repeat = np.tile(l_xyz, (N,1))              # (N*L, 3)
#     # offset ray origin along normal to prevent self-intersection (shadow acne) + offset along ray direction for larger acceptable window of visibility
#     origins = pts_repeat + (normal_repeat * eps) + (dir_repeat * radius)      # ray tracing from surface point to light direction

#     vis_flat = np.zeros(N*L, dtype=bool)

#     for start in range(0, N*L, chunk_size):
#         end = min(start + chunk_size, N*L)
#         rays = mi.Ray3f(
#             o=mi.Point3f(origins[start:end, 0], origins[start:end, 1], origins[start:end, 2]),
#             d=mi.Vector3f(dir_repeat[start:end, 0], dir_repeat[start:end, 1], dir_repeat[start:end, 2])
#         )
#         rays.maxt = dr.inf  # max position on ray segment set to infinity

#         occluded = np.array(scene.ray_test(rays))   # returns True if ray hits something ==> not visible
#         vis_flat[start:end] = ~occluded             # visibility set to opposite of occluded 

#     vis = vis_flat.reshape(N,L)
#     vis[invalid] = True
#     return vis


## for point light sources
def compute_visibility(scene, surface_pts, normals, light_pos, eps=0.03, chunk_size=1 << 20, disable=False):
    """
    Computes visibility of each point, but each ray terminates at the light positions (finite distance). 
    """
    if disable:
        # Disable visibility, all points are visible
        N, L = surface_pts.shape[0], light_pos.shape[0]
        return np.ones((N,L), dtype=bool)

    normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    invalid = (normal_norm.squeeze(-1) < 1e-5)
    normals = normals / (normal_norm + 1e-8)
    if invalid.any():
        normals[invalid] = np.array([0.0, 0.0, 1.0], dtype=normals.dtype)

    N, L = surface_pts.shape[0], light_pos.shape[0]
    pts_r = np.repeat(surface_pts, L, axis=0)
    norm_r = np.repeat(normals, L, axis=0)
    light_r = np.tile(light_pos, (N,1))

    to_light = light_r - pts_r
    dist = np.linalg.norm(to_light, axis=-1, keepdims=True)
    dirs = to_light / np.clip(dist, 1e-8, None)
    origins = pts_r + norm_r * eps + dirs * eps 

    vis_flat = np.zeros(N*L, dtype=bool)
    for start in range(0, N*L, chunk_size):
        end = min(start + chunk_size, N*L)
        rays = mi.Ray3f(
            o=mi.Point3f(origins[start:end, 0], origins[start:end, 1], origins[start:end, 2]),
            d=mi.Vector3f(dirs[start:end, 0],dirs[start:end, 1], dirs[start:end, 2])
        )
        rays.maxt = dist[start:end, 0] * 0.995
        occluded = np.array(scene.ray_test(rays))
        vis_flat[start:end] = ~occluded

    vis = vis_flat.reshape(N,L)
    vis[invalid] = True
    return vis


################################################################################################################################################################
# Auxiliary Mesh Anchoring
################################################################################################################################################################

from scipy.spatial import cKDTree

def build_aux_mesh_tree(aux_mesh_path, norm_scale=1.0, norm_center=0.0):
    """
    Loads auxiliary mesh, so that the trained SDF doesn't deform excessively.
    Done through a KD-Tree to query nearest-neighbour of points.
    """

    if isinstance(aux_mesh_path, (str, Path)):
        aux_mesh = trimesh.load(aux_mesh_path, process=False)
        aux_verts = np.asarray(aux_mesh.vertices, dtype=np.float32)
    else:
        aux_verts = np.asarray(aux_mesh_path, dtype=np.float32)

    norm_center_arr = np.asarray(norm_center, dtype=np.float32).reshape(1, 3) \
                       if np.ndim(norm_center) > 0 else np.float32(norm_center)

    aux_verts = (aux_verts - norm_center_arr) / norm_scale

    tree = cKDTree(aux_verts)
    aux_verts_t = torch.from_numpy(aux_verts).float()
    return tree, aux_verts_t, aux_mesh.faces, aux_mesh.vertex_normals
    

def aux_mesh_loss(surface_pts, aux_tree, aux_verts_t, max_dist=None, k_candidates=8):
    """
    Compute the loss as distance between points on the trained SDF and auxiliary mesh.

    Args:
        surface_pts:        (N,3) surface points on current SDF
        aux_tree:           KD-Tree representing the auxiliary mesh
        aux_verts_t:        (M,3) vertices on auxiliary mesh as torch tensor
        max_dist:           Drop correspondences when the distance between compared points exceeds this threshold.

    Returns:
        mean L2 distance from surface points to auxiliary vertex    
    """

    device = surface_pts.device
    N = surface_pts.shape[0]
    
    # STEP 1: KD-tree finds k nearest neighbors (CPU)
    surface_pts_np = surface_pts.detach().cpu().numpy()
    _, idxs = aux_tree.query(surface_pts_np, k=k_candidates, workers=-1)
    # idxs shape: (N, k) where N=16384, k=8
    
    # STEP 2: Move ALL candidates to GPU at once
    idxs = torch.from_numpy(idxs).long().to(device)  # (16384, 8)
    aux_verts_t = aux_verts_t.to(device)
    candidates = aux_verts_t[idxs]                   # (16384, 8, 3)
    
    # STEP 3: Compute ALL distances at once on GPU
    surf_expanded = surface_pts.unsqueeze(1)         # (16384, 1, 3)
    dist = (surf_expanded - candidates).norm(dim=-1) # (16384, 8)
    best_dist = dist.min(dim=1)[0]                   # (16384,)
    
    return best_dist.mean()


################################################################################################################################################################
# Point Lights Visualization + Light Source Detection
################################################################################################################################################################

import matplotlib.pyplot as plt

def project_light_position_to_image(light_pos, pose, fl_x, fl_y, cx, cy, height, width, scene_scale=1.0):
    """
    Project a 3D point light source position onto the image plane for visualization purposes.

    Args:
        light_pos:          (L,3) world-space light positions
        pose:               (4,4) camera-to-world matrix
        fl_x, fl_y, cx, cy: camera intrinsics

    Returns:
        (L,2) pixel coords & (L,) boolean mask of points that are in front of the camera
    """        

    pose_np = pose.detach().cpu().numpy() if torch.is_tensor(pose) else pose
    rot, trans = pose_np[:3,:3], pose_np[:3,3]

    # transform from world into camera space
    cam_pts = (light_pos - trans) @ rot

    # points that lie in front of camera
    in_front = cam_pts[:, 2] < 0

    # pinhole projection
    x_ndc = cam_pts[:, 0] / (-cam_pts[:, 2] + 1e-8)
    y_ndc = cam_pts[:, 1] / (-cam_pts[:, 2] + 1e-8)

    col = cx + x_ndc * fl_x
    row = cy - y_ndc * fl_y

    pix = np.stack([col, row], axis=-1)
    return pix, in_front


def plot_light_positions(image, light_pos_world, intensities, pose, fl_x, fl_y, cx, cy, height, width, ax=None):
    """
    Plot the point light sources on given image.
    """

    pix, in_front = project_light_position_to_image(light_pos_world, pose, fl_x, fl_y, cx, cy, height, width)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))

    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    for i in range(len(pix)):
        if not in_front[i]: # skip if light source does not lie in front of camera
            continue

        col, row = pix[i]

        if 0 <= col < width and 0 <= row < height:
            size = 100 + intensities[i].mean() * 400
            ax.scatter(col, row, s=size, c=[np.clip(intensities[i], 0, 1)], edgecolors='black', linewidths=1.5, marker='*')
    ax.set_title("Point Light Sources Positions")
    return ax


def draw_lights(image, light_pos_world, intensities, pose,
                fl_x, fl_y, cx, cy, height, width,
                top_k=5, dpi=100):
    
    pix, in_front = project_light_position_to_image(
        light_pos_world, pose, fl_x, fl_y, cx, cy, height, width
    )

    inten = np.asarray(intensities)
    brightness = inten if inten.ndim == 1 else inten.mean(axis=-1)

    # Select top-K brightest valid lights
    valid = np.where(in_front)[0]
    order = valid[np.argsort(-brightness[valid])]
    top_idx = order[:top_k]

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])         # full-bleed, no margins
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    for i in top_idx:
        col, row = pix[i]
        if 0 <= col < width and 0 <= row < height:
            size = 100 + brightness[i] * 400
            ax.scatter(col, row, s=size,
                       c=[np.clip(intensities[i], 0, 1)] if np.ndim(intensities[i]) else 'red',
                       edgecolors='black', linewidths=1.5, marker='*')

    # Render to a numpy array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())     # HxWx4 uint8
    plt.close(fig)
    return buf[..., :3]                            # drop alpha


def export_lights_with_mesh(mesh_path_or_mesh, light_positions, out_path,
                            light_radius=0.02, light_color=(255, 60, 0, 255),
                            base_mesh_color=(200, 200, 200, 255)):
    """
    Merges point light source positions (rendered as small spheres) with a mesh
    and exports the combined result as a single PLY for visual debugging.

    Args:
        mesh_path_or_mesh: path to a mesh file, or an already-loaded trimesh.Trimesh
        light_positions:   (L,3) numpy array of light positions, already in world/scene scale
        out_path:          output path for the combined debug mesh
        light_radius:      radius of the sphere marker for each light
        light_color:       RGBA colour for light spheres (default: orange)
        base_mesh_color:   RGBA colour to force onto the base mesh if it has none

    Returns:
        The combined trimesh.Trimesh object (also exported to out_path).
    """
    if isinstance(mesh_path_or_mesh, (str, Path)):
        base_mesh = trimesh.load(mesh_path_or_mesh, process=False)
    else:
        base_mesh = mesh_path_or_mesh.copy()

    # Ensure base mesh has vertex colours so concatenation doesn't fail/mismatch
    vc = getattr(base_mesh.visual, "vertex_colors", None)
    if vc is None or len(vc) != len(base_mesh.vertices):
        base_mesh.visual.vertex_colors = np.tile(
            base_mesh_color, (len(base_mesh.vertices), 1)
        ).astype(np.uint8)

    light_meshes = []
    for pos in np.asarray(light_positions):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=light_radius)
        sphere.apply_translation(pos)
        sphere.visual.vertex_colors = np.tile(
            light_color, (len(sphere.vertices), 1)
        ).astype(np.uint8)
        light_meshes.append(sphere)

    combined = trimesh.util.concatenate([base_mesh] + light_meshes)
    combined.export(out_path)
    return combined


def light_pos_loss(point_lights, aux_tree, aux_verts_t, eps=1e-3):
    """
    Computes the loss for penalization of light sources that situate themselves close to a surface.
    This pushes light sources away from surfaces/reflections.
    """ 
    device = point_lights.pos_raw.device
    positions = point_lights.positions().to(device)
    positions_np = positions.detach().cpu().numpy()

    _, idx = aux_tree.query(positions_np, k=1, workers=-1)
    idx_t = torch.from_numpy(idx).long().to(device)
    aux_verts_t = aux_verts_t.to(device)
    nearest_pts = aux_verts_t[idx_t]

    d_near = (positions - nearest_pts).norm(dim=-1).clamp(min=eps)

    active = point_lights.active_mask.to(device)
    l_pos = (active * torch.exp(-100.0 * d_near)).sum() / active.sum().clamp(min=1.0)

    return l_pos


def light_attraction_loss(point_lights, emitter_positions, emitter_weights, temperature=5.0):
    """
    Pulls point light sources towards surface points that are bright/near-saturated to locate possible light/emissive sources.

    Args:
        point_lights:           Point Light Sources
        emitter_positions:      Positions of true emitters in the scene
        emitter_weights:        Weights of emitters
        temperature:            Controls distribution of light based on distance from light source (low: soft, diffuse attraction || high: sharp, focused attraction)
        hover_offset:           Offset attraction targets, so that point lights are attracted to the front of the surface, not on it
    """
    device = point_lights.pos_raw.device

    if emitter_positions.shape[0] == 0:
        print("No emitters detected in the scene!")
        return torch.tensor(0.0, device=device)
    
    light_pos = point_lights.positions()                # (L,3)
    dist = torch.cdist(light_pos, emitter_positions)    # (L,M)
    soft_w = F.softmax(-dist * temperature + torch.log(emitter_weights.unsqueeze(0) + 1e-8), dim=-1)
    attraction = (soft_w * dist).sum(dim=-1)

    active = point_lights.active_mask.to(device)
    return (attraction * active).sum() / active.sum().clamp(min=1.0)


################################################################################################################################################################
# Point Light Initialization - IID
################################################################################################################################################################

def farthest_point_sampling(points, n_samples, weights=None):
    """
    Greedy farthest point sampling: Iteratively pick the farthest point from already-selected set for a well-spread out final subset.
    """

    M = points.shape[0]
    if M <= n_samples:
        return np.arange(M)

    if weights is None:
        weights = np.ones(M)
    weights = weights / (weights.sum() + 1e-8)  # normalize

    selected = []
    first = int(np.argmax(weights)) if weights is not None else 0
    selected.append(first)

    dist = np.linalg.norm(points - points[first], axis=-1)  # (M,)

    for _ in range(n_samples - 1):
        # pick point that lies farthest from already selected point, based on brightness
        score = dist * weights
        score[selected] = -1
        next_idx = int(np.argmax(score))
        selected.append(next_idx)
        new_dist = np.linalg.norm(points - points[next_idx], axis=-1)
        dist = np.minimum(dist, new_dist)

    return np.array(selected)


def backproject_light(renderer, images_t, poses_t, init_raydir,
                      idx_train, height, width, device, scene_scale,
                      grid_step=16, n_views=40, normal_offset=0.02,
                      bright_thresh=0.8, appearance_embedding=None, near_far_fn=None, chunk_size=8192, is_vanilla=False):
    """
    Grid-based light candidate generation, based on Intrinsic Image Diffusion.
    For a regular grid of pixels across several training views, backproject to SDF surface and offset along normal.

    Args:
        grid_step:      pixel stride of sampling grid
        n_views:        number of training images to grid over
        normal_offset:  distance to offset the candidate off the surface
        bright_thresh:  candidates are weighted by pixel brightness
        near_far_fn:    get_near_far_bounds_box

    Returns:
        candidates:     (M,3) numpy array of world-space candidate positions
        weights:        (M,)  numpy array of per-candidate brightness weights
    """

    assert near_far_fn is not None, "use get_near_far_bounds_box as near_far_fn"

    # Pick spread views of training images over the scene
    view_choice = np.linspace(0, len(idx_train) - 1, n_views).astype(int)
    view_ids = idx_train[view_choice]

    # Build pixel grid
    rows = torch.arange(0, height, grid_step, device=device)
    cols = torch.arange(0, width, grid_step, device=device)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    grid_rows, grid_cols = grid_rows.reshape(-1), grid_cols.reshape(-1)

    grid_rows_cpu, grid_cols_cpu = grid_rows.cpu(), grid_cols.cpu()

    all_candidates = []
    all_weights = []

    for img_idx in view_ids:
        pose = poses_t[img_idx].to(device)
        rot, trans = pose[:3, :3], pose[:3, 3]

        cam_dirs = init_raydir[grid_rows, grid_cols]    # (G,3)
        world_dirs = torch.einsum("ij,gj->gi", rot, cam_dirs)
        world_dirs = F.normalize(world_dirs, dim=-1)
        world_origs = (trans / scene_scale).expand(world_dirs.shape)

        # Ground truth brightness at grid pixels
        gt_pixels = images_t[img_idx, grid_rows_cpu, grid_cols_cpu].to(device) # (G,3)
        brightness = 0.2126 * gt_pixels[:,0] + 0.7152 * gt_pixels[:,1] + 0.0722 * gt_pixels[:,2]    # (G,)

        if appearance_embedding is not None:
            appearance_code = appearance_embedding(torch.tensor([img_idx], device=device))
        else:
            appearance_code = None

        n_rays = world_dirs.shape[0]
        for start in range(0, n_rays, chunk_size):
            end = start + chunk_size
            ro = world_origs[start:end]
            rd = world_dirs[start:end]
            near, far = near_far_fn(ro, rd)
            near, far = near.to(device), far.to(device)

            if is_vanilla:
                with torch.enable_grad():
                    surface_pts, normals, weight_sum, _, _, _ = compute_surface_pts_and_normals(renderer, rd, ro, near, far, scene_scale, appearance_embedding=appearance_code)
                
            else:
                with torch.no_grad():
                    surface_pts, normals, weight_sum, _, _, _ = compute_surface_pts_and_normals(renderer, rd, ro, near, far, scene_scale, appearance_embedding=appearance_code)

            valid = weight_sum > 0.5
            if valid.sum() == 0:
                continue

            surf_valid = surface_pts[valid]
            norm_valid = F.normalize(normals[valid], dim=-1)
            bright_valid = brightness[start:end][valid]

            candidate_pts = surf_valid + norm_valid * normal_offset

            all_candidates.append(candidate_pts.detach().cpu().numpy())
            all_weights.append(bright_valid.detach().cpu().numpy())

    candidates = np.concatenate(all_candidates, axis=0)
    weights = np.concatenate(all_weights, axis=0)

    weights = np.clip(weights, bright_thresh * 0.3, None)   # bias weights towards bright grids

    return candidates, weights


def initialize_point_lights(point_lights, emitter_positions, emitter_weights, 
                            fallback_candidates, fallback_weights, 
                            scene_bounds=1.0, jitter_std=0.01):
    """
    Seed the first K point lights to predicted emitter positions and 
    select remaining n_lights spread-out candidates using farthest point sampling and 
    writes them into point_lights.pos_raw for initialization.
    """

    n_lights = point_lights.n_lights
    device = point_lights.pos_raw.device
    K = min(len(emitter_positions), n_lights)

    seeded = emitter_positions[:K].copy()
    seeded = seeded + np.random.normal(scale=jitter_std, size=seeded.shape) # add jitter so lights don't start in duplicate spots

    remaining = n_lights - K
    if remaining > 0:
        fb_idx = farthest_point_sampling(fallback_candidates, remaining, weights=fallback_weights)
        seeded = np.concatenate([seeded, fallback_candidates[fb_idx]], axis=0)

    normed = np.clip(seeded / scene_bounds, -0.995, 0.995)
    pos_raw_init = np.arctanh(normed)

    with torch.no_grad():
        point_lights.pos_raw[:] = torch.from_numpy(pos_raw_init).float().to(device)

    print(f"Initialized {K}/{n_lights} point lights from detected emitters, the remaining {remaining} from fallback candidates.")


def detect_emitters_multiview(renderer, images_t, poses_t, init_raydir,
                               idx_train, height, width, device, scene_scale,
                               aux_tree, aux_verts_t, aux_normals, near_far_fn,
                               grid_step=8, bright_thresh=0.99, prefilter_thresh=0.5,
                               min_views=10, max_std=0.05, chunk_size=8192,
                               appearance_embedding=None, normal_offset=0.05, is_vanilla=False):
    """
    Detect true emitters that appear consistently from all training images.
    """

    n_verts = aux_verts_t.shape[0]
    sum_bright = torch.zeros(n_verts, device=device)
    sum_sq     = torch.zeros(n_verts, device=device)
    view_count = torch.zeros(n_verts, device=device)

    rows = torch.arange(0, height, grid_step, device=device)
    cols = torch.arange(0, width, grid_step, device=device)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    grid_rows, grid_cols = grid_rows.reshape(-1), grid_cols.reshape(-1)

    grid_rows_cpu, grid_cols_cpu = grid_rows.cpu(), grid_cols.cpu()

    aux_verts_t = aux_verts_t.to(device)

    if is_vanilla:
        with torch.enable_grad():
            for img_idx in idx_train:
                pose = poses_t[img_idx].to(device)
                rot, trans = pose[:3, :3], pose[:3, 3]

                cam_dirs = init_raydir[grid_rows, grid_cols]
                world_dirs = F.normalize(torch.einsum("ij,gj->gi", rot, cam_dirs), dim=-1)
                world_origs = (trans / scene_scale).expand(world_dirs.shape)

                gt_pixels = images_t[img_idx, grid_rows_cpu, grid_cols_cpu].to(device)
                brightness = 0.2126*gt_pixels[:,0] + 0.7152*gt_pixels[:,1] + 0.0722*gt_pixels[:,2]

                # cheap prefilter -- skip clearly-dim regions before running the SDF renderer
                keep = brightness > prefilter_thresh
                if keep.sum() == 0:
                    continue

                wd, wo, br = world_dirs[keep], world_origs[keep], brightness[keep]
                appearance_code = (appearance_embedding(torch.tensor([img_idx], device=device))
                                if appearance_embedding is not None else None)

                for start in range(0, wd.shape[0], chunk_size):
                    end = start + chunk_size
                    ro, rd = wo[start:end], wd[start:end]
                    near, far = near_far_fn(ro, rd)
                    near, far = near.to(device), far.to(device)

                    surface_pts, normals, weight_sum, _, _, _ = compute_surface_pts_and_normals(
                        renderer, rd, ro, near, far, scene_scale, appearance_embedding=appearance_code)

                    valid = weight_sum > 0.5
                    if valid.sum() == 0:
                        continue

                    pts_np = surface_pts[valid].detach().cpu().numpy()
                    _, vidx = aux_tree.query(pts_np, k=1, workers=-1)
                    vidx_t = torch.from_numpy(vidx).long().to(device)
                    b_chunk = br[start:end][valid]

                    sum_bright.index_add_(0, vidx_t, b_chunk)
                    sum_sq.index_add_(0, vidx_t, b_chunk ** 2)
                    view_count.index_add_(0, vidx_t, torch.ones_like(b_chunk))

    else:
        with torch.no_grad():
            for img_idx in idx_train:
                pose = poses_t[img_idx].to(device)
                rot, trans = pose[:3, :3], pose[:3, 3]

                cam_dirs = init_raydir[grid_rows, grid_cols]
                world_dirs = F.normalize(torch.einsum("ij,gj->gi", rot, cam_dirs), dim=-1)
                world_origs = (trans / scene_scale).expand(world_dirs.shape)

                gt_pixels = images_t[img_idx, grid_rows_cpu, grid_cols_cpu].to(device)
                brightness = 0.2126*gt_pixels[:,0] + 0.7152*gt_pixels[:,1] + 0.0722*gt_pixels[:,2]

                # cheap prefilter -- skip clearly-dim regions before running the SDF renderer
                keep = brightness > prefilter_thresh
                if keep.sum() == 0:
                    continue

                wd, wo, br = world_dirs[keep], world_origs[keep], brightness[keep]
                appearance_code = (appearance_embedding(torch.tensor([img_idx], device=device))
                                if appearance_embedding is not None else None)

                for start in range(0, wd.shape[0], chunk_size):
                    end = start + chunk_size
                    ro, rd = wo[start:end], wd[start:end]
                    near, far = near_far_fn(ro, rd)
                    near, far = near.to(device), far.to(device)

                    surface_pts, normals, weight_sum, _, _, _ = compute_surface_pts_and_normals(
                        renderer, rd, ro, near, far, scene_scale, appearance_embedding=appearance_code)

                    valid = weight_sum > 0.5
                    if valid.sum() == 0:
                        continue

                    pts_np = surface_pts[valid].detach().cpu().numpy()
                    _, vidx = aux_tree.query(pts_np, k=1, workers=-1)
                    vidx_t = torch.from_numpy(vidx).long().to(device)
                    b_chunk = br[start:end][valid]

                    sum_bright.index_add_(0, vidx_t, b_chunk)
                    sum_sq.index_add_(0, vidx_t, b_chunk ** 2)
                    view_count.index_add_(0, vidx_t, torch.ones_like(b_chunk))

    mean_b = sum_bright / view_count.clamp(min=1)
    std_b  = (sum_sq / view_count.clamp(min=1) - mean_b**2).clamp(min=0).sqrt()

    emitter_mask = (mean_b > bright_thresh) & (std_b < max_std) & (view_count >= min_views)

    emitter_positions = aux_verts_t[emitter_mask]
    emitter_weights = mean_b[emitter_mask]

    aux_normals_t = torch.from_numpy(np.array(aux_normals, copy=True)).float().to(device)
    emitter_positions = emitter_positions + aux_normals_t[emitter_mask] * normal_offset

    print(f"Detected {emitter_positions.shape[0]} emitter vertices "
          f"(out of {n_verts}, seen bright & consistent across ≥{min_views} views).")
    return emitter_positions, emitter_weights


def cluster_emitter_points(positions, weights, merge_dist=0.05):
    """
    Greedy single-linkage clustering of detected emitter vertices into distinct
    fixtures. Processes points in descending weight (brightness) order so each
    cluster's seed is its brightest member.

    Args:
        positions: (M,3) numpy array of detected emitter positions
        weights:   (M,) numpy array of brightness weights
        merge_dist: points within this distance of a cluster's running centroid
                    are absorbed into it

    Returns:
        cluster_positions: (K,3) weighted centroid per cluster
        cluster_weights:   (K,) summed weight per cluster
        cluster_sizes:     (K,) number of points absorbed per cluster
    """
    order = np.argsort(-weights)
    positions, weights = positions[order], weights[order]

    assigned = np.full(len(positions), -1, dtype=int)
    centroids, csum_w, csizes = [], [], []

    tree = cKDTree(positions)
    for i in range(len(positions)):
        if assigned[i] != -1:
            continue
        cid = len(centroids)
        assigned[i] = cid
        nearby = tree.query_ball_point(positions[i], r=merge_dist)
        for j in nearby:
            if assigned[j] == -1:
                assigned[j] = cid

        members = assigned == cid
        w = weights[members]
        centroids.append((positions[members] * w[:, None]).sum(0) / w.sum())
        csum_w.append(w.sum())
        csizes.append(members.sum())

    return np.array(centroids), np.array(csum_w), np.array(csizes)


################################################################################################################################################################
# Weight annealing
################################################################################################################################################################

def anneal_weight(epoch, w_init, w_final, anneal_epochs=40000):
    """
    Anneal given weights during training.
    """
    t = min(epoch / anneal_epochs, 1.0)
    return w_init + (w_final - w_init) * t
