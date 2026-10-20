import matplotlib.pyplot as plt
import numpy as np
import torch 
import neus_helper as neuh
import neusfactor_helper as neusfach
import neusfactor_nets as neunet
import nerf_helper as nh
from torch import nn, optim
from torch.nn import functional as F
from pathlib import Path
import tinycudann as tcnn
import subprocess
import sys
import os
import trimesh
import datetime

# export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-19.so

now = datetime.datetime.now()
print(f"Current date & time of running script: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current run is NeuSFactor WITH material priors!")


#######################################################################################################################################################
# Script start
#######################################################################################################################################################
    

## Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently running on {device}...", flush=True)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Dataset object
obj = "6d89a7320d"  # 6d89a7320d, 40aec5fffa, bd9305480d, 5a269ba6fe, 08bbbdcc3d
model_path = f"models/neus_trained-{obj}-500k.pth"

os.makedirs(f"result/neus/{obj}", exist_ok=True)
os.makedirs("models", exist_ok=True)


## Hyperparameters
chunk_size = 1024 * 32  # orig 1024 * 32

# Number of rays for training iteration
batch_image_size = 128   # orig 128
pixel_batch = batch_image_size ** 2

# Refresh mesh and reload Mitsuba scene interval
remesh_interval = 10000
mesh_resolution = 512
 

scannetdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scannet'))
basedir = f"{scannetdir}/data/{obj}"

os.makedirs(f"result/scannet/{obj}", exist_ok=True)
os.makedirs("models", exist_ok=True)

# in NeuS renderer, scene scale is expected to be [-1,1]^3
SCENE_SCALE = 1.0   ## for ScanNet++


## Prepare data
# Import & prepare data
images, depths, alphas, poses, hw, intrinsics, i_split, norm_center, norm_scale, image_ids_split = nh.load_scannetpp_data(basedir, skip_bad=True) 
height, width = hw
fl_x, fl_y, cx, cy = intrinsics["fl_x"], intrinsics["fl_y"], intrinsics["cx"], intrinsics["cy"]
idx_train, idx_val, idx_test = i_split
train_ids, val_ids, test_ids = image_ids_split
print(f"Loaded {len(images)} images, H={height}, W={width}", flush=True)
print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}", flush=True)
print(f"Debug for Image ID Split: | Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}", flush=True)
print(f"Focal length x : {fl_x} | Focal length y : {fl_y} | cx : {cx} | cy : {cy}")


## Pinhole
cols = torch.arange(width)
rows = torch.arange(height)
cent_x = cols - cx
cent_y = cy - rows      # y is flipped as image coords y is opposite direction

# Create meshgrid using xy-indexing
cent_x, cent_y = torch.meshgrid(cent_x, cent_y, indexing="xy") 

# Create 3D pixel coords
pixel_coords = torch.stack([cent_x / fl_x, cent_y / fl_y, torch.full_like(cent_x, -1.0)], dim=-1)   # Z coordinate set to negative of focal length

# Setup initial ray direction 
init_raydir = pixel_coords.to(device)          # shape: H, W, 3(xyz)


## Debug
# init_raydir has shape (H, W, 3). The pixel at exactly (cx, cy) should have
# a ray direction of (0, 0, -1) in camera space, since that's the definition
# of the principal point -- straight ahead.
row_center = int(round(cy))
col_center = int(round(cx))
center_ray = init_raydir[row_center, col_center]
print("Ray direction at principal point (should be ~[0,0,-1]):", center_ray)


# Convert to tensors
images_t = torch.from_numpy(images).float() #.to(device) # (N,H,W,3)
alphas_t = torch.from_numpy(alphas).float() #.to(device) # (N,H,W)
poses_t = torch.from_numpy(poses).float().to(device)   # (N,4,4)
depths_t = torch.from_numpy(depths).float() #.to(device) # (N,H,W) in metres, normalized


## Load material priors - albedo, metallic, roughness
alb_prior, met_prior, rou_prior = neusfach.load_material_priors(obj, image_ids_split, height, width)
alb_prior_t = torch.from_numpy(alb_prior).float() #.to(device)   # (N,H,W,3)
met_prior_t = torch.from_numpy(met_prior).float() #.to(device)   # (N,H,W,1)
rou_prior_t = torch.from_numpy(rou_prior).float() #.to(device)   # (N,H,W,1)
print(f"Loaded material priors: albedo {alb_prior_t.shape}, metallic {met_prior_t.shape}, roughness {rou_prior_t.shape}", flush=True)


## Models Setup
d_appearance = 16
appearance_embedding = nn.Embedding(len(images), d_appearance).to(device)   # nn.Embedding is a lookup table
nn.init.normal_(appearance_embedding.weight, mean=0.0, std=0.01)    # start near zero

sdf_network = neuh.HashSDFNetwork(d_feature=256, bias=0.5, inside_outside=True).to(device)
sdf_network.set_lambda(sdf_network.n_levels)

deviation_network = neuh.SingleVarianceNetwork(init_val=0.3).to(device)

colour_network = neuh.RenderingNetwork(d_feature=256, mode="idr", d_in=9, d_out=3, d_hidden=256, n_layers=4,
                                       weight_norm=True, multires_view=4, squeeze_out=True, d_appearance=d_appearance).to(device)

## Depths regularization
depth_scale_raw = nn.Parameter(torch.tensor(0.0, device=device))   # passed through softplus for positivity

# Load frozen trained INGP NeuS checkpoints 
checkpoint = torch.load(model_path, map_location=device)

sdf_network.load_state_dict(checkpoint["sdf_network"])
deviation_network.load_state_dict(checkpoint["deviation_network"])
colour_network.load_state_dict(checkpoint["color_network"])
appearance_embedding.load_state_dict(checkpoint["appearance_embedding"])
depth_scale_raw = nn.Parameter(checkpoint["depth_scale_raw"].to(device))

# Joint SDF Fine-Tuning
# Allow SDF and deviation networks to train
sdf_network.train()
for p in sdf_network.parameters():
    p.requires_grad_(True)

deviation_network.train()
for p in deviation_network.parameters():
    p.requires_grad_(True)

colour_network.eval()
for p in colour_network.parameters():
    p.requires_grad_(False)

appearance_embedding.eval()
for p in appearance_embedding.parameters():
    p.requires_grad_(False)

renderer = neuh.NeuSRenderer(nerf=None, sdf_network=sdf_network, deviation_network=deviation_network, color_network=colour_network,
                             n_samples=64, n_importance=64, n_outside=0, up_sample_steps=4, perturb=1.0, compute_curvature=False)   


## Mesh extraction for Mitsuba visibility
mesh_dir = f"result/{obj}/mesh"
os.makedirs(mesh_dir, exist_ok=True)
mesh_path = f"{mesh_dir}/nodepth_sdf_neusfactor_mesh_{obj}.ply"

# Define the bounding box for extraction
bound_min = torch.tensor([-1.0, -1.0, -1.0], device=device)  
bound_max = torch.tensor([1.0, 1.0, 1.0], device=device)     
resolution = 512  # Higher = more detail, but slower

# Extract mesh vertices and triangles
vertices, triangles = renderer.extract_geometry(
    bound_min=bound_min,
    bound_max=bound_max,
    resolution=resolution,
    threshold=0.0  # Surface is where SDF = 0
)

print(f"Extracting mesh ...")
mesh = trimesh.Trimesh(vertices=vertices * SCENE_SCALE, faces=triangles)
mesh.export(mesh_path)
print(f"Saved mesh successfully at {mesh_path}!")
print(f"Mesh has bounds: {mesh.bounds}")

mitsuba_scene = neusfach.load_mitsuba_scene(mesh_path)
print("Loaded Mitsuba scene successfully.\n")


## Load auxiliary mesh
aux_mesh_path = f"{basedir}/scans/mesh_aligned_0.05.ply"
aux_tree, aux_verts_t, aux_faces, aux_vert_normals = neusfach.build_aux_mesh_tree(aux_mesh_path, norm_scale=norm_scale, norm_center=norm_center)

## Debug
aux_verts_np = aux_verts_t.detach().cpu().numpy()
transformed_aux_mesh = trimesh.Trimesh(vertices=aux_verts_np, faces=aux_faces, process=False)
transformed_aux_mesh.export(f"{mesh_dir}/auxiliary_mesh_test_{obj}.ply")


# ## Generate light directions and solid angles
# light_height = 16
# l_xyz, areas = neusfach.get_lights_and_angles(height=light_height)
# l_xyz = l_xyz.to(device)
# areas = areas.to(device)
# L = l_xyz.shape[0]
# print(f"Using {L} light directions ({light_height}x{light_height*2} environment map)")


## Initialize network components in NeRFactor
neusfactor_mlp = neunet.NeuSFactorMLPs().to(device)
# env_light = neunet.EnvironmentLight(height=light_height).to(device)
point_lights = neunet.PointLightSources(n_lights=48).to(device)
ambient_light = neunet.AmbientLight(init_val=0.05).to(device)


## Load NeuSFactor checkpoint
# ckpt_path = f"models/ingpneusfactor_trained-{obj}.pth"
# ckpt = torch.load(ckpt_path, map_location=device)
# neusfactor_mlp.load_state_dict(ckpt["neusfactor_mlp"])
# env_light.load_state_dict(ckpt["env_light"])
# point_lights.load_state_dict(ckpt["point_lights"])
# ambient_light.load_state_dict(ckpt["ambient_light"])



## Detect emitter pre-training loop
emitter_raw_positions, emitter_raw_weights = neusfach.detect_emitters_multiview(renderer, images_t, poses_t, init_raydir,
                                                                        idx_train, height, width, device, SCENE_SCALE,
                                                                        aux_tree, aux_verts_t, aux_vert_normals, neuh.get_near_far_bounds_box,
                                                                        appearance_embedding=appearance_embedding)

# Cluster emitters
cluster_pos, cluster_w, cluster_sizes = neusfach.cluster_emitter_points(emitter_raw_positions.detach().cpu().numpy(), emitter_raw_weights.detach().cpu().numpy(), merge_dist=0.1)


print(f"Cluster positions: \n {cluster_pos} \n")

## Remove outliers from detected emitters
UP_AXIS = 2  # z-coords

positive_z_mask = cluster_pos[:, UP_AXIS] > 0.0     # keep emitters that are at top part of mesh
n_below = np.sum(~positive_z_mask)
print(f"Filtered {n_below} emitter(s) with Z <= 0 out of {cluster_pos.shape[0]} initially detected emitters.")

cluster_pos = cluster_pos[positive_z_mask]
cluster_w = cluster_w[positive_z_mask]
cluster_sizes = cluster_sizes[positive_z_mask]

heights = cluster_pos[:, UP_AXIS]
median_h = np.median(heights)
mad = np.median(np.abs(heights - median_h)) + 1e-6

keep_mask = np.abs(heights - median_h) < 4.0 * mad

print(f"Filtered {np.sum(~keep_mask)} outlier(s) out of the {cluster_pos.shape[0]} initially detected emitters.")

cluster_pos = cluster_pos[keep_mask]
cluster_w = cluster_w[keep_mask]
cluster_sizes = cluster_sizes[keep_mask]


emitter_positions = torch.from_numpy(cluster_pos).float().to(device)
emitter_weights = torch.from_numpy(cluster_w).float().to(device)
print(f"Using {emitter_positions.shape[0]} fixed emitter targets for attraction loss.")


## Debug Emitter Visualization
# Create spheres for bright points (same as lights)
emitter_np = emitter_positions.detach().cpu().numpy() * SCENE_SCALE
emitter_spheres = []
for pos in emitter_np:
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.015)
    sphere.apply_translation(pos)
    sphere.visual.vertex_colors = np.tile([255, 255, 0, 255], (len(sphere.vertices), 1))
    emitter_spheres.append(sphere)

fallback_candidates, fallback_weights = neusfach.backproject_light(renderer, images_t, poses_t, init_raydir,
                                                                   idx_train, height, width, device, SCENE_SCALE, 
                                                                   appearance_embedding=appearance_embedding, near_far_fn=neuh.get_near_far_bounds_box)
neusfach.initialize_point_lights(point_lights, emitter_np, emitter_weights.detach().cpu().numpy(), fallback_candidates, fallback_weights)


## Debug point lights
debug_lights_path = f"{mesh_dir}/nodepth_{obj}_initial_light_pos.ply"
init_light_pos = point_lights.positions().detach().cpu().numpy() * SCENE_SCALE
neusfach.export_lights_with_mesh(mesh_path, init_light_pos, debug_lights_path,
                                 light_radius=0.02, light_color=(255, 60, 0, 255),
                                 base_mesh_color=(200, 200, 200, 255))
print(f"Saved debug mesh with initial point light positions to {debug_lights_path}", flush=True)


# Load current mesh and lights
mesh_with_lights = trimesh.load(mesh_path)
light_pos_np = point_lights.positions().detach().cpu().numpy() * SCENE_SCALE
light_spheres = []
for pos in light_pos_np:
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    sphere.apply_translation(pos)
    sphere.visual.vertex_colors = np.tile([255, 60, 0, 255], (len(sphere.vertices), 1))
    light_spheres.append(sphere)

combined = trimesh.util.concatenate([mesh_with_lights] + light_spheres + emitter_spheres)
combined.export(f"{mesh_dir}/{obj}_emitter_detection.ply")
print(f"Saved debug emitter visualization in {mesh_dir}/{obj}_emitter_detection.ply!")



## Optimizer
lr = 5e-4
env_lr = 1e-3
sdf_lr = 5e-4
dev_lr = 5e-4
optimizer = optim.Adam([
    {"params": list(neusfactor_mlp.parameters()), "lr": lr},
    # {"params": list(env_light.parameters()), "lr": env_lr},
    {"params": list(point_lights.parameters()), "lr": env_lr},
    {"params": list(ambient_light.parameters()), "lr": 1e-3},
    {"params": list(sdf_network.parameters()), "lr": sdf_lr},
    {"params": list(deviation_network.parameters()), "lr": dev_lr},
    {"params": [depth_scale_raw], "lr":lr}  
])


epochs = 100001
pixel_batch = 128 ** 2


## Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


## Initiate model training
pixel_count = height * width
pixel_prob = torch.full((pixel_count,), 1/pixel_count).to(device)   # create tensor with uniform probability of a pixel getting selected for a batch of rays
xyz_jitter_std = 0.01   # orig 0.01
max_grad_norm = 1.0

# Accumulate results
epoch_list = []
PSNR_list = []
best_val_loss = float("inf")

visibility = None

## Setup for stagnation, point light pruning
stagnation_patience = 5000
stagnation_counter = 0
best_stagnation_loss = float('inf')
prune_rel_thresh = 0.05

## Weights

# Light-related
w_light_sparsity = 1e-4
w_light_pos = 1e-3
w_attract = 5e-2
w_light_final = 1e-6

# Priors
w_prior_alb = 0.05
w_prior_met = 0.005
w_prior_rou = 0.005

# Depth
w_depth = 0.01

# Auxiliary
w_aux_mesh = 1.0


## Training + test loop
for epoch in range(epochs):
    image_idx = np.random.choice(idx_train)
    rays_o, rays_d, target, mask, rows, cols, depth_target = neuh.sample_rays(images_t, alphas_t, poses_t, init_raydir, image_idx, pixel_batch, height, width, device, SCENE_SCALE, depths_t=depths_t)
    rays_o, rays_d, target, mask, depth_target = rays_o.to(device), rays_d.to(device), target.to(device), mask.to(device), depth_target.to(device)

    appearance_code = appearance_embedding(torch.tensor([image_idx], device=device))

    gt_mask = mask.squeeze(-1) > neusfach.GT_ALPHA_THRESH
    if gt_mask.sum() == 0:
        continue

    rows, cols = rows.long(), cols.long()
    rows_cpu, cols_cpu = rows.cpu(), cols.cpu()

    alb_prior_batch = alb_prior_t[image_idx, rows_cpu, cols_cpu].to(device)
    met_prior_batch = met_prior_t[image_idx, rows_cpu, cols_cpu].to(device)
    rou_prior_batch = rou_prior_t[image_idx, rows_cpu, cols_cpu].to(device)

    rays_o, rays_d, target = rays_o[gt_mask], rays_d[gt_mask], target[gt_mask]
    target_alpha_batch = mask.squeeze(-1)[gt_mask]
    depth_batch = depth_target[gt_mask]
    alb_prior_batch = alb_prior_batch[gt_mask]
    met_prior_batch = met_prior_batch[gt_mask]
    rou_prior_batch = rou_prior_batch[gt_mask]

    near, far = neuh.get_near_far_bounds_box(rays_o, rays_d)
    near, far = near.to(device), far.to(device)

    surface_pts, normals, weight_sum, gradients, sampled_pts, peak_depth = neusfach.compute_surface_pts_and_normals(renderer, rays_d, rays_o, near, far, SCENE_SCALE, appearance_embedding=appearance_code)
        
    if torch.isnan(normals).any():
        print(f"Epoch {epoch}: NaN in normals, skipping ...\n")
        continue

    # Depth
    depth_scale = F.softplus(depth_scale_raw) + 1e-4
    
    # NeuSFactor normal pass
    albedo_pred, metallic_pred, roughness_pred = neusfactor_mlp(surface_pts)
    view_dirs = F.normalize(-rays_d, dim=-1)

    # Visibility prediction
    surface_pts_np = surface_pts.detach().cpu().numpy()
    normals_np = normals.detach().cpu().numpy()
    light_pos_np = point_lights.positions().detach().cpu().numpy()

    vis_np = neusfach.compute_visibility(mitsuba_scene, surface_pts_np, normals_np, light_pos_np, disable=False)    # visib toggle
    visibility = torch.tensor(vis_np, dtype=torch.float32, device=device)

    # C_r = neusfach.render_neusfactor(view_dirs, normals, albedo_pred,
    #                                  metallic_pred, roughness_pred, env_light,
    #                                  l_xyz, areas, visibility=visibility)
    C_r = neusfach.render_neusfactor(surface_pts, view_dirs, normals,
                                     albedo_pred, metallic_pred, roughness_pred, 
                                     point_lights, ambient_light, visibility=visibility)
    C_r = C_r * target_alpha_batch.unsqueeze(-1) + (1.0 - target_alpha_batch.unsqueeze(-1)) * 1.0

    # Jittered 
    surface_pts_jit = surface_pts + torch.randn_like(surface_pts) * xyz_jitter_std
    albedo_jit, metallic_jit, roughness_jit = neusfactor_mlp(surface_pts_jit)


    # Get weights
    current_w_light_sparsity = neusfach.anneal_weight(epoch, w_light_sparsity, w_light_final)
    current_w_light_pos = neusfach.anneal_weight(epoch, w_light_pos, w_light_final)
    current_w_attract = neusfach.anneal_weight(epoch, w_attract, w_light_final)

    current_w_prior_alb = neusfach.anneal_weight(epoch, w_prior_alb, 0.01, anneal_epochs=45000)
    current_w_prior_met = neusfach.anneal_weight(epoch, w_prior_met, 0.001, anneal_epochs=45000)
    current_w_prior_rou = neusfach.anneal_weight(epoch, w_prior_rou, 0.001, anneal_epochs=45000)
    
    current_w_depth = 0.0 # neusfach.anneal_weight(epoch, w_depth, 0.01, anneal_epochs=45000)
    current_w_aux_mesh = neusfach.anneal_weight(epoch, w_aux_mesh, 0.1, anneal_epochs=45000)
    
    if epoch % 5000 == 0:
        print(f"Current weights || Light sparsity: {current_w_light_sparsity:.6f} || Light position: {current_w_light_pos:.6f} || Attraction: {current_w_attract:.6f}")
        print(f"Current weights || Albedo prior: {current_w_prior_alb:.6f} || Metallic prior: {current_w_prior_met:.6f} || Roughness prior: {current_w_prior_rou:.6f}")
        print(f"Current weights || Depth: {current_w_depth:.6f} || Auxiliary: {current_w_aux_mesh:.6f}")


    # Loss calculation
    loss, loss_dict = neusfach.neusfactor_loss(C_r, target, albedo_pred, albedo_jit,
                                               metallic_pred, metallic_jit, roughness_pred, roughness_jit, point_lights,
                                               gradients, sampled_pts, surface_pts, aux_tree, aux_verts_t, emitter_positions, emitter_weights, 1.0, 
                                               depth_scale, depth_batch, peak_depth,
                                               weight_sum, target_alpha_batch,
                                               alb_prior_batch, met_prior_batch, rou_prior_batch,
                                               w_light_sparsity=current_w_light_sparsity, w_light_pos=current_w_light_pos, w_attract=current_w_attract,
                                               w_depth=current_w_depth, w_prior_alb=current_w_prior_alb, w_prior_met=current_w_prior_met, w_prior_rou=current_w_prior_rou,
                                               w_aux_mesh=current_w_aux_mesh)

    # Debug: Print loss components
    if epoch % 5000 == 0:
        print(f"Epoch {epoch}:")
        for k, v in loss_dict.items():
            if np.isnan(v) or np.isinf(v):
                print(f"  {k}: {v} <-- PROBLEM!")
            else:
                print(f"  {k}: {v:.6f}")
        print(f"  Alpha coverage: {target_alpha_batch.mean():.3f}")
        print("")

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping to prevent NaN
    torch.nn.utils.clip_grad_norm_(list(neusfactor_mlp.parameters()) + 
                                   list(point_lights.parameters()) +
                                   list(ambient_light.parameters()) +
                                   list(sdf_network.parameters()) +
                                   list(deviation_network.parameters()) +
                                   [depth_scale_raw],
                                   max_grad_norm)

    optimizer.step()

    # Learning rate decay
    scheduler.step()


    # Remeshing for Mitsuba scene
    if epoch > 0 and epoch % remesh_interval == 0:
        print(f"Epoch {epoch} : Remeshing Mitsuba scene and saving it...")
        remesh_path = f"{mesh_dir}/nodepth_sdf_mesh_{obj}_epoch_{epoch}.ply"    # name
        out_light_path = f"{mesh_dir}/nodepth_light_position_mesh_{obj}_epoch_{epoch}.ply"
        light_positions = point_lights.positions().detach().cpu().numpy() * SCENE_SCALE
        vertices, triangles, mitsuba_scene = neusfach.remesh_scene(renderer, bound_min, bound_max, mesh_resolution, SCENE_SCALE, remesh_path, light_positions, out_light_path)


    # Validation loop
    # Save image of progress every ? epoch
    # Saves checkpoint for best validation loss
    if epoch % 5000 == 0:
        neusfactor_mlp.eval()
        point_lights.eval()
        sdf_network.eval()
        deviation_network.eval()

        # # Environment light map
        # env_map = env_light.map.detach().cpu().squeeze(0).permute(1,2,0)    # (1,3,H,W) --> (H,W,3)
        # env_map_disp = env_map / (env_map.max() + 1e-8)


        with torch.no_grad():
            losses_dict, mean_psnr, C_r_val, alb_val, met_val, rou_val, normal_val, vis_val, random_pick = neusfach.evaluate_neusfactor(renderer, neusfactor_mlp, point_lights, ambient_light,
                                                                                                                                        images_t[idx_val], alphas_t[idx_val], poses_t[idx_val], depths_t[idx_val], depth_scale, init_raydir,
                                                                                                                                        height, width, SCENE_SCALE, device, idx_val, aux_tree, aux_verts_t, emitter_positions, emitter_weights,
                                                                                                                                        xyz_jitter_std, max_images=3, mitsuba_scene=mitsuba_scene, appearance_embedding=appearance_embedding,
                                                                                                                                        alb_prior_t=alb_prior_t[idx_val], met_prior_t=met_prior_t[idx_val], rou_prior_t=rou_prior_t[idx_val], disable_visib=False, # visib toggle
                                                                                                                                        current_w_light_sparsity=current_w_light_sparsity, current_w_light_pos=current_w_light_pos, current_w_attract=current_w_attract,
                                                                                                                                        current_w_depth=current_w_depth, current_w_prior_alb=current_w_prior_alb, current_w_prior_met=current_w_prior_met, current_w_prior_rou=current_w_prior_rou,
                                                                                                                                        current_w_aux_mesh=current_w_aux_mesh)   

        
        # Point light sources
        light_pos_val = point_lights.positions().detach().cpu().numpy() * SCENE_SCALE
        light_int_val = point_lights.intensities().detach().cpu().numpy()
        val_pose = poses_t[idx_val[random_pick]].detach().cpu().numpy()

        print(f"Validation losses: ")
        for k, v in losses_dict.items():
            if np.isnan(v) or np.isinf(v):
                print(f"  {k}: {v} <-- PROBLEM!")
            else:
                print(f"  {k}: {v:.6f}")
        print("")

        with torch.no_grad():
            total_int = point_lights.intensities().sum(dim=-1)  # (12,)
            sorted_int, order = torch.sort(total_int, descending=True)
            print(f"Point light intensities in order: {sorted_int.cpu().numpy()}")


        ## Pruning if loss is stagnating
        if losses_dict["total"] < best_stagnation_loss * (1 - 0.005):
            best_stagnation_loss = losses_dict["total"]
            stagnation_counter = 0
        else:
            stagnation_counter += 1000

        if stagnation_counter >= stagnation_patience:
            print(f"Epoch {epoch} : Loss is stagnating. Proceeding with point light source merging and pruning...")
            n_merged, _ = point_lights.merge_close_lights(merge_dist=0.05)
            print(f"Merged {n_merged} co-located lights.")
            n_pruned, n_active = point_lights.prune_weak_lights(rel_thresh=prune_rel_thresh)
            print(f"Successfully pruned {n_pruned} point lights. There are only {n_active} out of {point_lights.n_lights} point lights remaining.")
            stagnation_counter = 0


        # Save checkpoint for best validation loss
        if losses_dict["total"] < best_val_loss:
            print("Found new best validation loss!", flush=True)
            best_val_loss = losses_dict["total"]
            model_name = f"nodepth_ingpneusfactor_trained-{obj}.pth" # name
            model_save_path = f"models/{model_name}"
            print(f"Saving checkpoint models to {model_save_path} ...", flush=True)
            torch.save({
                "neusfactor_mlp" : neusfactor_mlp.state_dict(),
                "point_lights" : point_lights.state_dict(),
                "ambient_light" : ambient_light.state_dict(),
                "sdf_network" : sdf_network.state_dict(),
                "deviation_network" : deviation_network.state_dict(),
                "depth_scale_raw" : depth_scale_raw.detach().cpu()
            }, model_save_path)
            print("Saved successfully.\n", flush=True)

        # Peak Signal to Noise Ratio
        PSNR_list.append(torch.tensor(mean_psnr))
        epoch_list.append(epoch)

        epochs_plot = [e.item() if torch.is_tensor(e) else e for e in epoch_list]
        psnr_plot = [p.item() if torch.is_tensor(p) else p for p in PSNR_list]

        # Plot progress in terms of view and PSNR & save image of plot
        fig, axes = plt.subplots(3,3, figsize=(24,16))
        axes[0,0].imshow(images[idx_val[random_pick]])
        axes[0,0].set_title("Ground Truth")
        axes[0,1].imshow(C_r_val.detach().cpu().numpy())
        axes[0,1].set_title(f"Rendered RGB with PSNR {mean_psnr.item():.2f}")

        neusfach.plot_light_positions(images[idx_val[random_pick]], light_pos_val, light_int_val, val_pose, 
                             fl_x, fl_y, cx, cy, height, width, ax=axes[0,2])

        axes[1,0].imshow(alb_val.detach().cpu().numpy())
        axes[1,0].set_title("Albedo")
        axes[1,1].imshow((normal_val.detach().cpu().numpy() + 1.0) / 2.0)
        axes[1,1].set_title("SDF Normals")
        axes[1,2].imshow(met_val.detach().cpu().numpy())
        axes[1,2].set_title("Metallic")
        axes[2,0].imshow(rou_val.detach().cpu().numpy())
        axes[2,0].set_title("Roughness")
        axes[2,1].imshow(vis_val.detach().cpu().numpy())
        axes[2,1].set_title("Visibility")
        axes[2,2].plot(epochs_plot, psnr_plot)
        axes[2,2].set_title("PSNR Progression")
        axes[2,2].set_xlabel("Epochs")
        axes[2,2].set_ylabel("PSNR")

        # Hide the axis ticks
        for ax in axes.flat:
            ax.axis('off')

        axes[2,2].axis('on')

        plt.tight_layout()
        plt.suptitle(f"Epoch {epoch}")
        filename = f"result/{obj}/nodepth_neusfactor_{obj}_epoch_{epoch}"   # name
        plt.savefig(filename)
        plt.close()

        # switch back to train mode
        neusfactor_mlp.train()
        point_lights.train()
        sdf_network.train()
        deviation_network.train()

        print(f"Complete {epoch}th epoch! Saved plot to '{filename}.png'. \n", flush=True)

print("", flush=True)
print("Training complete!", flush=True)


## Export inverse rendered mesh

neusfactor_mlp.eval()
point_lights.eval()
ambient_light.eval()
sdf_network.eval()
deviation_network.eval()

print("Re-extracting final mesh from fine-tuned SDF ...", flush=True)
vertices, triangles = renderer.extract_geometry(
    bound_min=bound_min, bound_max=bound_max, resolution=resolution, threshold=0.0
)

with torch.no_grad():
    query_pts = torch.from_numpy(vertices).float().to(device) * SCENE_SCALE

    albedo_list, metallic_list, roughness_list = [], [], []
    for i in range(0, query_pts.shape[0], chunk_size):
        chunk = query_pts[i:i+chunk_size]
        alb, met, rou = neusfactor_mlp(chunk)
        albedo_list.append(alb.cpu())
        metallic_list.append(met.cpu())
        roughness_list.append(rou.cpu())

    albedo_full = torch.cat(albedo_list, dim=0).numpy()
    metallic_full = torch.cat(metallic_list, dim=0).numpy()
    roughness_full = torch.cat(roughness_list, dim=0).numpy()

albedo_rgb = np.clip(albedo_full, 0.0, 1.0)
vertex_colours = (albedo_rgb * 255).astype(np.uint8)
if vertex_colours.shape[1] == 3:
    alpha_col = np.full((vertex_colours.shape[0], 1), 255, dtype=np.uint8)
    vertex_colours = np.concatenate([vertex_colours, alpha_col], axis=1)    # RGBA

inv_mesh = trimesh.Trimesh(vertices=vertices * SCENE_SCALE,
                           faces=triangles,
                           vertex_colors=vertex_colours,
                           process=False)

inv_mesh_path = f"{mesh_dir}/nodepth_neusfactor_inv_mesh_{obj}.ply"
inv_mesh.export(inv_mesh_path)
print(f"Saved inverse-rendered mesh to {inv_mesh_path}!")

end = datetime.datetime.now()
print(f"Current end date & time of finishing script: {end.strftime('%Y-%m-%d %H:%M:%S')}")
