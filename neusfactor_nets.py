import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tinycudann as tcnn
from neusfactor_helper import normalize_coords

## Helper functions --------------------------------------------------------------------------------------------

def l2_normalize(v, dim=-1, eps=1e-6):
    return v / torch.sqrt(torch.sum(v ** 2, dim=dim, keepdim=True) + eps)

# TinyCudaNN Config Builders

def hashgrid_config(n_levels=16,
                    n_features=2,
                    base_resolution=16,
                    per_level_scale=1.5) -> dict:
    """
    Returns dict with tcnn hash grid structure.
    """

    return {
        "otype" : "HashGrid",
        "n_levels" : n_levels,
        "n_features_per_level" : n_features,
        "log2_hashmap_size" : 19,
        "base_resolution" : base_resolution,
        "per_level_scale" : per_level_scale,
        "interpolation" : "Linear",
    }

def fusedmlp_config(n_neurons=64,
                    n_hidden=2,
                    activation="ReLU",
                    out_activation="None") -> dict:

    return{
        "otype" : "FullyFusedMLP",
        "activation" : activation,
        "output_activation" : out_activation,
        "n_neurons" : n_neurons,
        "n_hidden_layers" : n_hidden,
    }


## MLPs ---------------------------------------------------------------------------------------------------------


# Network that combines all NeRFactor MLP components
class NeuSFactorMLPs(nn.Module):
    """
    NeuSFactor Network that combines all MLP components under a single class.
    All component MLPs share a single hash-grid encoder.
    """
    ROUGHNESS_MIN = 0.04    # avoid GGX singularity at roughness=0

    def __init__(self, n_levels=16, n_features=2, n_neurons=64, n_hidden=3):
        super().__init__()
        
        # shared hash grid 
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=hashgrid_config(n_levels, n_features)
        )
        enc_dims = self.encoder.n_output_dims   # n_levels * n_features 32 by default

        # albedo
        self.albedo_head = tcnn.Network(
            n_input_dims=enc_dims,
            n_output_dims=3,
            network_config=fusedmlp_config(n_neurons, n_hidden, out_activation="None")
        )

        # brdf
        self.brdf_head = tcnn.Network(
            n_input_dims=enc_dims,
            n_output_dims=2,
            network_config=fusedmlp_config(n_neurons, n_hidden, out_activation="None")
        )
    
    # encode/decode helper functions
    def _encode(self, xcoords):
        norm_xcoords = normalize_coords(xcoords).clamp(0.001, 0.999)
        pos_encoding = self.encoder(norm_xcoords)
        return torch.clamp(pos_encoding, -100.0, 100.0)

    def _get_albedo(self, pos_encoding):
        return torch.sigmoid(self.albedo_head(pos_encoding).float())

    def _get_brdf(self, pos_encoding):
        out = self.brdf_head(pos_encoding).float()
        metallic = torch.sigmoid(out[:,0:1])
        roughness = torch.sigmoid(out[:,1:2]) * (1.0 - self.ROUGHNESS_MIN) + self.ROUGHNESS_MIN # manual normalization
        return metallic, roughness

    def forward(self, xcoords):
        """
        Runs all MLP heads simultaneously.
        """
        pos_encoding  = self._encode(xcoords)
        albedo = self._get_albedo(pos_encoding)
        metallic, roughness = self._get_brdf(pos_encoding)
        return albedo, metallic, roughness

    def albedo(self, xcoords):
        pos_encoding  = self._encode(xcoords)
        albedo = self._get_albedo(pos_encoding)
        return albedo

    def brdf(self, xcoords):
        pos_encoding  = self._encode(xcoords)
        metallic, roughness = self._get_brdf(pos_encoding)
        return metallic, roughness


# Environment Light Map, L -- Blender Synthetic dataset
class EnvironmentLight(nn.Module):
    """
    Learnable HDR environment map in latitude-longitude format, L.
    
    Default of 16x32 resolution.
    """

    def __init__(self, height=16, init_max=0.5):
        super().__init__()
        self.height = height
        self.width = height * 2
        self.map = nn.Parameter(torch.rand(1, 3, height, self.width) * init_max)

    def forward(self):
        return torch.clamp(self.map, min=0) # ensure non-negative value

    def gradient_penalty_spatial(self):
        """
        Spatial total variation penalty: ||∂L/∂u||² + ||∂L/∂v||².
        """
        L = self.map    # shape (1, 3, H, W)
        dx = L - torch.roll(L, shifts=1, dims=3) # horizontal diff
        dy = L - torch.roll(L, shifts=1, dims=2) # vertical diff
        return (dx ** 2).mean() + (dy ** 2).mean()

    def gradient_penalty_chromatic(self):
        """
        Cross-channel total variation penalty: ||∂L/∂channel||².
        """
        L = self.map  
        dc = L - torch.roll(L, shifts=1, dims=1)  # channel difference
        return (dc ** 2).mean()

    def white_balance_penalty(self):
        """
        Light regularization assuming near-natural white incident light.
        Penalizes deviation of each colour channel from the mean across the channels.
        """
        L = self.map    # (1,3,H,W)
        mean_rgb = L.mean(dim=1, keepdim=True)  # (1,1,H,W)
        diff = L - mean_rgb
        return (diff ** 2).sum(dim=1).mean()   # sum over RGB, average over pixels


## Replacement for Light Environment Map -- Indoor Scenes
class PointLightSources(nn.Module):
    """
    For an indoor scene (e.g. ScanNet++), N learnable point light sources are trained with position and RGB intensity.
    Training converges into the actual total number of light sources in the scene.
    Unused light sources are driven towards 0 intensity.
    Each point light has a Spherical Gaussian emission profile, based on IntrinsicImageDiffusion paper.
    """

    def __init__(self, n_lights=12, scene_bounds=1.0, 
                 init_intensity=0.3, max_intensity=5.0, 
                 init_sharpness_raw=-5.0, max_sharpness=16.0):
        super().__init__()
        self.n_lights = n_lights
        self.scene_bounds = scene_bounds
        self.max_intensity = max_intensity
        self.max_sharpness = max_sharpness

        self.pos_raw = nn.Parameter(torch.randn(n_lights,3) * 0.7)                  # random init positions within bounding box
        self.intensity_raw = nn.Parameter(torch.full((n_lights,3), init_intensity)) # intensity set to initial value

        # random initial axes
        self.axis_raw = nn.Parameter(torch.randn(n_lights, 3))
        self.sharpness_raw = nn.Parameter(torch.full((n_lights,), init_sharpness_raw))  # initialize near-isotropic 

        # pruning mask
        self.register_buffer("active_mask", torch.ones(n_lights))   

    def positions(self):
        # positions are kept within bounding box
        return torch.tanh(self.pos_raw) * self.scene_bounds

    def intensities(self):
        raw = F.softplus(self.intensity_raw)
        return torch.clamp(raw, max=self.max_intensity)  * self.active_mask.unsqueeze(-1)
        # return self.max_intensity * torch.sigmoid(self.intensity_raw) * self.active_mask.unsqueeze(-1)

    def axes(self):
        return F.normalize(self.axis_raw, dim=-1)

    def sharpness(self):
        return F.softplus(self.sharpness_raw).clamp(max=self.max_sharpness)

    def emission(self, directions):
        """
        Arg:
            directions:     (N,L,3) unit vectors that point from the light source towards queried points

        Returns:
            (N,L,3) RGB emitted radiance per light
        """
        xi = self.axes()        # (L,3)
        lam = self.sharpness()  # (L,)
        mu = self.intensities() # (L,3)

        cos_term = (directions * xi.unsqueeze(0)).sum(-1)       # (N,L)
        lobe = torch.exp(lam.unsqueeze(0) * (cos_term - 1.0))   # (N,L) in (0,1]
        return mu.unsqueeze(0) * lobe.unsqueeze(-1)             # (N,L,3)

    def sparsity_loss(self):
        # lights that do not contribute to reconstruction are driven to 0 intensity
        return self.intensities().sum(dim=-1).mean()

    @torch.no_grad()
    def prune_weak_lights(self, rel_thresh=0.2):
        """
        Adaptive pruning of point light sources as described in Intrinsic Image Diffusion paper.
        Any light whose total intensity is below the rel_thresh (default 5% - current 20%) of the current strongest intensity will be pruned.
        Pruning is done only when the loss value stagnates. 
        """

        total_intensity = self.intensities().sum(dim=-1)  # (L,)
        active_intensity = total_intensity * self.active_mask
        strongest = active_intensity.max()

        if strongest <= 0:
            return 0, int(self.active_mask.sum().item())

        pruned = (total_intensity < rel_thresh * strongest) & (self.active_mask > 0)
        self.active_mask[pruned] = 0.0
        n_pruned = int(pruned.sum().item())
        n_active = int(self.active_mask.sum().item())

        return n_pruned, n_active

    @torch.no_grad()
    def merge_close_lights(self, merge_dist=0.05):
        """
        Merge light sources that are closer than merge_dist.
        The brighter light absorbs the dimmer one's intensity.
        """
        positions = self.positions()                         # (L,3)
        intensities = self.intensities()                     # (L,3)
        total_int = intensities.sum(dim=-1)                  # (L,)
        active = self.active_mask.clone()
        n_merged = 0
        for i in range(self.n_lights):
            if active[i] == 0:
                continue
            for j in range(i + 1, self.n_lights):
                if active[j] == 0:
                    continue
                d = (positions[i] - positions[j]).norm()
                if d < merge_dist:
                    # Keep the brighter one, absorb the dimmer one
                    if total_int[i] >= total_int[j]:
                        keep, kill = i, j
                    else:
                        keep, kill = j, i
                    # Add killed light's intensity (in raw space, approximate)
                    combined = intensities[keep] + intensities[kill]
                    self.intensity_raw.data[keep] = torch.log(torch.exp(combined) - 1.0 + 1e-8)  # inverse softplus
                    active[kill] = 0.0
                    n_merged += 1
        self.active_mask.copy_(active)
        n_active = int(active.sum().item())
        return n_merged, n_active


class AmbientLight(nn.Module):
    """
    A learnable approximation of indirect illumination.
    """

    def __init__(self, init_val=0.05):
        super().__init__()
        self.ambient_raw = nn.Parameter(torch.full((3,), init_val))

    def get_value(self):
        return F.softplus(self.ambient_raw) # non-negative