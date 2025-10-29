import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.Pointfilter_Network_Architecture_3 import UNetDenoiser, RobustL1ChamferLoss, RepulsionLoss

from .renderer import NeRFRenderer
import numpy as np
from encoding import get_encoder
from .utils import safe_normalize
from implicit_neural_networks import IMLP


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=4,
                 hidden_dim=96,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 encoding='frequency_torch',
                 ):
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, multires=6)
        self.sigma_net = IMLP(self.in_dim, 4, hidden_dim, num_layers)
        self.sdf_net = IMLP(3, 1, hidden_dim, True, 6, [], 1, num_layers, geometric_init=True)

        # -----------------------------
        # 🔹 Integrate SUnet Denoiser
        # -----------------------------
        self.sunet = UNetDenoiser()  # adjust if you only have xyz or xyz+normal
        self.loss_chamfer = RobustL1ChamferLoss()
        self.loss_repulsion = RepulsionLoss()

        # Background model (unchanged)
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
        else:
            self.bg_net = None


    # ---------------------------------------------------------------
    # Common forward pass with SUnet denoising integration
    # ---------------------------------------------------------------
    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]
        h = self.encoder(x, bound=self.bound)
        h = self.sigma_net(h)
        sdf = self.sdf_net(x.view(-1, 3)).squeeze().view(x.shape[:-1])

        alpha = 0.1 * 1 / 0.001
        sigma = alpha * laplace_cumulative(-sdf, 0.001)
        albedo = torch.sigmoid(h[..., 1:])

        # -----------------------------
        # 🔹 Feed features to SUnet
        # -----------------------------
        # Combine xyz + sdf or xyz + albedo (6 channels total)
        combined = torch.cat([x, albedo], dim=-1).unsqueeze(0)  # [1, N, 6]
        combined = combined.permute(0, 2, 1)  # [B, C, N]
        refined = self.sunet(combined)  # UNetDenoiser forward
        refined = refined.permute(0, 2, 1).squeeze(0)  # back to [N, 6]

        # replace albedo with refined RGB or geometric correction
        refined_albedo = torch.sigmoid(refined[..., 3:])
        return sigma, sdf, refined_albedo


    # ---------------------------------------------------------------
    # Normal, forward, and density remain mostly same
    # ---------------------------------------------------------------
    def normal(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            sigma, sdf, albedo = self.common_forward(x)
            normal = -torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0]
        return safe_normalize(normal)


    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        if shading == 'albedo':
            sigma, sdf, color = self.common_forward(x)
            normal = None
        else:
            with torch.enable_grad():
                x.requires_grad_(True)
                sigma, sdf, albedo = self.common_forward(x)
                normal = -torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0]
            normal = safe_normalize(normal)
            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0)
            color = albedo * lambertian.unsqueeze(-1)
        return sigma, color, normal


    def density(self, x):
        sigma, sdf, albedo = self.common_forward(x)
        return {'sigma': sigma, 'albedo': albedo, 'sdf': sdf}


    def background(self, d):
        h = self.encoder_bg(d)
        h = self.bg_net(h)
        return torch.sigmoid(h)


    def get_params(self, lr):
        params = [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.sdf_net.parameters(), 'lr': lr},
            {'params': self.sunet.parameters(), 'lr': lr * 0.5},  # optional: lower LR for pretrained SUnet
        ]
        if self.bg_radius > 0:
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        return params


    # ---------------------------------------------------------------
    # 🔹 Optional: geometric regularization (Chamfer + Repulsion)
    # ---------------------------------------------------------------
    def regularization_loss(self, pred_points, target_points):
        loss_c = self.loss_chamfer(pred_points, target_points)
        loss_r = self.loss_repulsion(pred_points)
        return loss_c + 0.05 * loss_r
