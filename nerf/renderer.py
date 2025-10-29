import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .utils import safe_normalize
from unet.Pointfilter_Network_Architecture_3 import UNetDenoiser


# ---------- PDF Sampling for NeRF ----------
def sample_pdf(bins, weights, n_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


# ---------- Near / Far computation ----------
@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    radius = rays_o.norm(dim=-1, keepdim=True)
    if type == 'sphere':
        near = radius - bound
        far = radius + bound
    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15)
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        near = torch.clamp(near, min=min_near)
    return near, far


# ---------- Main Renderer ----------
class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh
        self.bg_radius = opt.bg_radius

        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # 🔹 Initialize S-UNet Denoiser
        # Input: 6D (xyz + predicted sigma or sdf)
        self.sunet = UNetDenoiser()

    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not hasattr(self, 'cuda_ray') or not self.cuda_ray:
            return
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    # ---------- RUN RENDER ----------
    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None,
            ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device
        results = {}

        aabb = self.aabb_train if self.training else self.aabb_infer
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)

        if light_d is None:
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0).expand((N, num_steps))
        z_vals = nears + (fars - nears) * z_vals

        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])

        # ---------- Density Query ----------
        density_outputs = self.density(xyzs.reshape(-1, 3))

        # ---------- Apply S-UNet for 3D Denoising ----------
        # Combine xyz + predicted sigma into a 6D tensor for denoising
        with torch.no_grad():  # optional, can remove if training jointly
            xyz_flat = xyzs.reshape(1, -1, 3)                      # [1, N*T, 3]
            sigma_flat = density_outputs['sigma'].reshape(1, -1, 1)  # [1, N*T, 1]
            sunet_in = torch.cat([xyz_flat, sigma_flat.expand_as(xyz_flat)], dim=-1)  # [1, N*T, 6]
            denoised_sigma = self.sunet(sunet_in)                  # [1, N*T, 1]
        density_outputs['sigma'] = denoised_sigma.squeeze(0).reshape(-1, 1)

        # ---------- Continue Normal Rendering ----------
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, 3)

        # ---------- Orientation / Smoothness ----------
        if normals is not None:
            normals = normals.view(N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.sum(-1).mean()
            normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2).view(N, -1, 3)
            loss_smooth = (normals - normals_perturb).abs()
            results['loss_smooth'] = loss_smooth.mean()

        weights_sum = weights.sum(dim=-1)
        depth = torch.sum(weights * z_vals, dim=-1)
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)

        if self.bg_radius > 0:
            bg_color = self.background(rays_d.reshape(-1, 3))
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyz'] = xyzs.reshape(-1, 3)

        return results

    # ---------- Render Wrapper ----------
    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        B, N = rays_o.shape[:2]
        device = rays_o.device
        if staged and (not hasattr(self, 'cuda_ray') or not self.cuda_ray):
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            weights_sum = torch.empty((B, N), device=device)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = self.run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    weights_sum[b:b+1, head:tail] = results_['weights_sum']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            results = {'depth': depth, 'image': image, 'weights_sum': weights_sum}
        else:
            results = self.run(rays_o, rays_d, **kwargs)
        return results
