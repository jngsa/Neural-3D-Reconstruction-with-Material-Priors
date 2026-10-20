import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
import tinycudann as tcnn


##########################################################################################################################################################################
# Embedder
##########################################################################################################################################################################


# Positional encoding embedding of NeRF. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs        # config params
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []                  # embedding functions
        d = self.kwargs['input_dims']   
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)   # identity function
            out_dim += d                    # add orig dimension to input

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # logarithmic spacing
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            # linear spacing
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:    # [torch.sin, torch.cos]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))    # for each freq, add sin(x*freq) and cos(x*freq)
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim      # total output = d * (1 + 2*N_freqs)

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


##########################################################################################################################################################################
# Fields
##########################################################################################################################################################################


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),  # layers where skip connections are added
                 multires=0,
                 bias=0.5,      # initial SDF bias
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]   # dims = [3, 256, 256, ..., 256, 257]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch  # replace input dimension with encoded dimension

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0] # skip connection
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:  # special weight initialization strategy for SDF
                if l == self.num_layers - 2:    # last layer
                    if not inside_outside:
                        # initiliaze last layer to approximate signed distance to a sphere
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:   # 1st layer with positional encoding
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0) # zero out encoded part
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:    # skip connection layers
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)    # zero out connections from pos. encoding
                else:   # standard initialization for other layers
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm: # normalizes weights
                lin = nn.utils.parametrizations.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs) # positional encoding

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:   # skip connection
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)  # softplus
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]   # return only sdf value

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):  # compute SDF gradients wrt input x (computes surface normals)
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True)[0]
        return gradients.unsqueeze(-1)


class HashSDFNetwork(nn.Module):
    def __init__(self,
                 d_hidden=256,
                 d_feature=256,
                 n_hidden_layers=1,
                 skip_layer=3,
                 bias=0.5,
                 n_levels=16,
                 n_features_per_level=2,
                 log2_hashmap_size=21,
                 base_resolution=32,
                 max_resolution=1024,   # prev 2048
                 softplus_beta=100.0,
                 inside_outside=False):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.current_lambda = None  # progressive training
        self.base_resolution = base_resolution
        self.per_level_scale = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1))
        
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype" : "HashGrid",
                "n_levels" : n_levels,
                "n_features_per_level" : n_features_per_level,
                "log2_hashmap_size" : log2_hashmap_size,
                "base_resolution" : base_resolution,
                "per_level_scale" : self.per_level_scale,
                "interpolation" : "Linear",
            }
        )

        ## not using tcnn Network due to softplus beta setting, weight norm and geometric init
        enc_dim = self.encoder.n_output_dims
        in_dim = enc_dim + 3
        
        ## Setup MLP
        self.n_hidden_layers = n_hidden_layers
        self.skip_layer = skip_layer
        self.d_hidden = d_hidden
        d_out = 1 + d_feature

        layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                lin = nn.Linear(in_dim, d_hidden)
            elif i == skip_layer:
                lin = nn.Linear(d_hidden + in_dim, d_hidden)
            else:
                lin = nn.Linear(d_hidden, d_hidden)
            layers.append(lin)

        out_layer = nn.Linear(d_hidden, d_out) 

        ## Geometric initialization
        for i, lin in enumerate(layers):
            nn.init.constant_(lin.bias, 0.0)
            
            if i == 0:
                nn.init.constant_(lin.weight[:, 3:], 0.0)
                nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(d_hidden))

            elif i == skip_layer:
                nn.init.normal_(lin.weight[:,:d_hidden], 0.0, np.sqrt(2) / np.sqrt(d_hidden))
                nn.init.normal_(lin.weight[:, d_hidden:d_hidden+3], 0.0, np.sqrt(2) / np.sqrt(d_hidden))
                nn.init.constant_(lin.weight[:, d_hidden+3:], 0.0)
            
            else:
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(d_hidden))

        if not inside_outside:  # camera trajectory is OUTSIDE the volume, looking IN
            nn.init.normal_(out_layer.weight, mean=np.sqrt(np.pi) / np.sqrt(d_hidden), std=1e-4)
            nn.init.constant_(out_layer.bias, -bias)
        else:                   # camera trajectory is INSIDE the volume, looking OUT
            nn.init.normal_(out_layer.weight, mean=-np.sqrt(np.pi) / np.sqrt(d_hidden), std=1e-4)
            nn.init.constant_(out_layer.bias, bias)

        # weight normalization
        for i in range(len(layers)):
            layers[i] = nn.utils.parametrizations.weight_norm(layers[i])
        out_layer = nn.utils.parametrizations.weight_norm(out_layer)

        self.layers = nn.ModuleList(layers)
        self.out_layer = out_layer
        self.activation = nn.Softplus(beta=softplus_beta)
        self.bias = bias

        self.eps_coarse = 2.0 / base_resolution
        self.eps_fine = 2.0 / max_resolution

    def set_lambda(self, lambda_val):
        self.current_lambda = lambda_val

    def get_coarse_to_fine_optim_mask(self, lambda_val: float) -> torch.Tensor:
        """
        Builds a mask for coarse to fine optimization by gradually increasing bandwidth of spatial grid encoding.
        The grid concatenates n_levels x n_features_per_level. (start at value of 2 [coarse] to 16 [fine])
        """

        F_ = self.n_features_per_level
        total = F_ * self.n_levels
        mask = torch.zeros(total)
        active_dims = min(int(lambda_val), self.n_levels) * F_
        mask[:active_dims] = 1.0
        return mask

    def current_eps(self):
        """
        Exponentially interpolate epsilon between coarse and fine bounds as lambda grows.
        """

        if self.current_lambda is None:
            return self.eps_fine

        t = min(self.current_lambda / self.n_levels, 1.0)
        return self.eps_coarse * (self.eps_fine / self.eps_coarse) ** t

    def forward(self, x):
        x_norm = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
        x_enc = self.encoder(x_norm).float()

        if self.current_lambda is not None and self.current_lambda < self.n_levels:
            mask = self.get_coarse_to_fine_optim_mask(self.current_lambda)
            mask = mask.to(device=x_enc.device, dtype=x_enc.dtype)
            x_enc = x_enc * mask

        h_in = torch.cat([x, x_enc], dim=-1)
        h = h_in

        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, h_in], dim=-1)
            h = self.activation(layer(h))

        out = self.out_layer(h)
        sdf = out[:,:1]
        feature = out[:,1:]
        return torch.cat([sdf, feature], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:,:1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):  # computes numerical gradient
        eps = self.current_eps()

        # define offset vectors, which are corners of tetrahedron in 3D space
        k1 = torch.tensor([1, -1, -1], device=x.device, dtype=x.dtype)
        k2 = torch.tensor([-1, -1, 1], device=x.device, dtype=x.dtype)
        k3 = torch.tensor([-1, 1, -1], device=x.device, dtype=x.dtype)
        k4 = torch.tensor([1, 1, 1], device=x.device, dtype=x.dtype)

        # evaluate sdf at each point
        sdf1 = self.sdf(x + k1 * eps)
        sdf2 = self.sdf(x + k2 * eps)
        sdf3 = self.sdf(x + k3 * eps)
        sdf4 = self.sdf(x + k4 * eps)

        # compute gradient using tetrahedral finite difference approx
        grad = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
        return grad.unsqueeze(-1)

    def curvature(self, x, normal=None):
        """
        Mean curvature approximated via finite differences.
        """

        eps = self.current_eps()
        if normal is None:
            normal = F.normalize(self.gradient(x).squeeze(-1), dim=-1)

        rand_dir = F.normalize(torch.randn_like(x), dim=-1)
        x_shifted = x + eps * rand_dir
        normal_shifted = F.normalize(self.gradient(x_shifted).squeeze(-1), dim=-1)
        return (normal - normal_shifted).norm(dim=-1) 


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):  # predict RGB colour given surface point, normal, view direction and feature vector
    def __init__(self,
                 d_feature, # feature vector dim from SDFNetwork
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 d_appearance=0):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.d_appearance = d_appearance
        dims = [d_in + d_feature + d_appearance] + [d_hidden for _ in range(n_layers)] + [d_out]

        # positional encoding of view direction
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)   # replace dim with positional encoded view dir dim

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors, appearance_code=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        # mode determines the way input is concatenated
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        if self.d_appearance > 0 and appearance_code is not None:
            rendering_input = torch.cat([rendering_input, appearance_code], dim=-1)

        # Setup MLP
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SingleVarianceNetwork(nn.Module): # param that controls sharpness of SDF-to-density transformation
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)


##########################################################################################################################################################################
# Renderer
##########################################################################################################################################################################


def extract_fields(bound_min, bound_max, resolution, query_func):   # sample SDF on 3D grid for mesh extraction
    N = 64  # chunk size

    # create grids
    device = bound_min.device if hasattr(bound_min, 'device') else torch.device('cpu')
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs) # create meshgrid for chunk
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # query values of SDF for all points in chunk
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val   # place values in output array
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):  # extract 3D mesh from SDF network
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)   # use marching cubes to extract isosurface
    b_max_np = bound_max.detach().cpu().numpy() # grid coords --> world coords
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):    # inverse CDF method to extract samples from PDF distribution
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1, device=bins.device), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds, device=bins.device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 compute_curvature=True):
        self.nerf = nerf    # nerf for background rendering
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.compute_curvature = compute_curvature

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background using separate NeRF model
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # distance btw samples
        dists = torch.cat([dists, torch.tensor([sample_dist], device=z_vals.device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)    # query NeRF
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)    # volume rendering
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s -- hierarchical sampling
        """
        # get 3D points
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3

        inside_box = pts.abs().amax(dim=-1) < 1.0
        inside_box = inside_box[:, :-1] | inside_box[:, 1:]

        # radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)   
        # inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)  # sample inside sphere
        
        # compute SDF
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=rays_o.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_box

        # estimate SDF at boundaries
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        # convert to density using logistic density distribution
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size, device=z_vals.device)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,       # main rendering function
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    appearance_code=None):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # dists = torch.cat([dists, torch.tensor([sample_dist], device=z_vals.device).expand(dists[..., :1].shape)], -1)
        dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        # query SDF Network
        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze(-1) # normals
        normals = F.normalize(gradients, dim=-1)

        appearance_expanded = None
        if appearance_code is not None:
            appearance_expanded = appearance_code[:, None, :].expand(batch_size, n_samples, -1).reshape(-1, appearance_code.shape[-1])

        sampled_color = color_network(pts, normals, dirs, feature_vector, appearance_code=appearance_expanded).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3], device=z_vals.device))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        # convert SDF to CDF (density) using logistic density function
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0) # equation (13) in NeuS paper

        ## sphere
        # pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        # inside_sphere = (pts_norm < 1.0).float().detach()
        # relax_inside_sphere = (pts_norm < 1.2).float().detach()

        ## box
        box_val = pts.abs().amax(dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_box = (box_val < 1.0).float().detach()
        relax_inside_box = (box_val < 1.2).float().detach()   # same 20% relax margin as before, now per-axis

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_box + background_alpha[:, :n_samples] * (1.0 - inside_box)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_box[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_box)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=z_vals.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Depth -- weighted sum of ray param t
        depth = (weights * mid_z_vals[:, :weights.shape[-1]]).sum(dim=-1, keepdim=True)   # (batch_size, 1)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_box * gradient_error).sum() / (relax_inside_box.sum() + 1e-5)

        # Curvature loss
        if self.compute_curvature:
            normal = F.normalize(gradients, dim=-1)
            curvature = sdf_network.curvature(pts, normal=normal)
            curvature_error = (relax_inside_box.reshape(-1) * curvature).sum() / (relax_inside_box.sum() + 1e-5)
        else:
            curvature_error = torch.tensor(0.0, device=pts.device)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_box': inside_box,
            'curvature_error' : curvature_error,
            'depth' : depth,
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, appearance_code=None):
        batch_size = len(rays_o)

        # sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        sample_dist = (far - near) / self.n_samples
        
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=rays_o.device)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(rays_o.device)
            z_vals = z_vals + t_rand * sample_dist

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device=rays_o.device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']


        if appearance_code is not None:
            appearance_code = appearance_code.expand(batch_size, -1)

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    appearance_code=appearance_code)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_box': ret_fine['inside_box'],
            'mid_z_vals' : ret_fine['mid_z_vals'],
            'curvature_error': ret_fine['curvature_error'],
            'depth_fine': ret_fine['depth'], 
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))


##########################################################################################################################################################################
# Helper functions
##########################################################################################################################################################################


def get_near_far_bounds(rays_o, rays_d, radius=1.0):
    """
    Returns per-ray near/far bounds assuming the object sits within a sphere of given radius (default 1.0).
    """
    a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - radius
    far = mid + radius
    return near, far


def get_near_far_bounds_box(rays_o, rays_d, bound_min=-1.0, bound_max=1.0, eps=1e-8):
    """
    Returns per-ray near/far bounds assuming the camera origin lies within a cube [bound_min, bound_max]^3.
    """
    safe_d = torch.where(rays_d.abs() < eps, torch.full_like(rays_d, eps), rays_d)
    t1 = (bound_min - rays_o) / safe_d
    t2 = (bound_max - rays_o) / safe_d
    tmin = torch.minimum(t1,t2)
    tmax = torch.maximum(t1,t2)
    near = tmin.max(dim=-1, keepdim=True)[0]
    far = tmax.min(dim=-1, keepdim=True)[0]
    near = near.clamp(min=0.0)
    far = far.clamp(min=near + 1e-5)    # ensure near < far
    return near, far


def sample_rays(images_t, alphas_t, poses_t, init_raydir, image_idx, n_rays, height, width, device, scene_scale, depths_t=None):
    """
    Sample n_rays of random pixels from one training image.
    """
    pose = poses_t[image_idx].to(device)
    rot, trans = pose[:3,:3], pose[:3,3]

    raydir_grid = torch.einsum("ij,hwj->hwi", rot, init_raydir) # (H,W,3)

    pixel_idx = torch.randint(0, height * width, (n_rays,), device=device)
    rows, cols = pixel_idx // width, pixel_idx % width

    rays_d = raydir_grid[rows, cols]        # (N,3)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = trans.expand(rays_d.shape) / scene_scale

    rows_cpu, cols_cpu = rows.cpu(), cols.cpu()
    target = images_t[image_idx][rows_cpu, cols_cpu].to(device)             # (N,3)
    mask = alphas_t[image_idx][rows_cpu, cols_cpu].unsqueeze(-1).to(device) # (N,1)

    depth_target = None
    if depths_t is not None:
        depth_target = depths_t[image_idx][rows_cpu, cols_cpu].unsqueeze(-1).to(device)

    return rays_o, rays_d, target, mask, rows, cols, depth_target


def render_full_image(renderer, pose, init_raydir, height, width, scene_scale, device, chunk=1024, appearance_code=None, is_scannet=True):
    """
    Render full image for visualization purposes.
    """
    renderer.sdf_network.eval()
    renderer.deviation_network.eval()
    renderer.color_network.eval()

    rot, trans = pose[:3, :3].to(device), pose[:3, 3].to(device)
    raydir_grid = torch.einsum("ij,hwj->hwi", rot, init_raydir)          # (H, W, 3)
    rays_d_full = F.normalize(raydir_grid, dim=-1).reshape(-1, 3)
    rays_o_full = (trans / scene_scale).expand(rays_d_full.shape)

    n_rays = rays_o_full.shape[0]
    colour_out = torch.zeros(n_rays, 3, device=device)
    weight_sum_out = torch.zeros(n_rays, 1, device=device)
    eikonal_sum = 0.0
    curvature_sum = 0.0

    with torch.enable_grad():
        for i in range(0, rays_d_full.shape[0], chunk):
            ro, rd = rays_o_full[i:i+chunk], rays_d_full[i:i+chunk]
            if is_scannet:
                near, far = get_near_far_bounds_box(ro, rd)
            else:
                near, far = get_near_far_bounds(ro, rd)
            render_out = renderer.render(ro, rd, near, far,
                                         background_rgb=torch.ones(3, device=device),
                                         cos_anneal_ratio=1.0,
                                         appearance_code=appearance_code)
            
            n_chunk_rays = ro.shape[0]
            colour_out[i:i+chunk] = render_out["color_fine"].detach()
            weight_sum_out[i:i+chunk] = render_out["weight_sum"].detach().clip(1e-3, 1.0 - 1e-3)
            eikonal_sum += render_out["gradient_error"].item() * n_chunk_rays   # gradient error is computed per chunk
            curvature_sum += render_out["curvature_error"].item() * n_chunk_rays

    renderer.sdf_network.train()
    renderer.deviation_network.train()
    renderer.color_network.train()

    return colour_out.reshape(height, width, 3), weight_sum_out.reshape(height, width, 1), eikonal_sum / n_rays, curvature_sum / n_rays


def evaluate_neus(renderer, images_t, alphas_t, poses_t, init_raydir, idx_val,
                   height, width, scene_scale, device, igr_weight, mask_weight, curv_weight=None, n_images=3, appearance_code=None, is_scannet=True):
    """
    Evaluate current training by using validation set.
    """
    renderer.sdf_network.eval(); renderer.deviation_network.eval(); renderer.color_network.eval()
    losses = []
    colour_losses = []
    mse_losses = []
    val_idxs = idx_val[:3] # pick fixed 3 images for validation # np.random.choice(idx_val, size=min(n_images, len(idx_val)), replace=False)
    with torch.enable_grad():
        for idx in val_idxs:
            img, weight_sum_img, mean_eikonal, mean_curvature = render_full_image(renderer, poses_t[idx], init_raydir, height, width, scene_scale, device, appearance_code=appearance_code, is_scannet=is_scannet)
            target = images_t[idx].to(device)
            mask_target = alphas_t[idx].to(device)

            colour_loss = F.l1_loss(img, target).item()
            mse_loss = F.mse_loss(img, target).item()
            mask_loss = F.binary_cross_entropy(weight_sum_img.reshape(-1), mask_target.reshape(-1)).item() if mask_weight > 0 else 0.0
            if curv_weight is not None:
                total_loss = colour_loss + mask_loss * mask_weight + mean_eikonal * igr_weight + mean_curvature * curv_weight
            else:
                total_loss = colour_loss + mask_loss * mask_weight + mean_eikonal * igr_weight
            losses.append(total_loss)
            colour_losses.append(colour_loss)
            mse_losses.append(mse_loss)

    renderer.sdf_network.train(); renderer.deviation_network.train(); renderer.color_network.train()
    return np.mean(losses), np.mean(colour_losses), np.mean(mse_losses)

