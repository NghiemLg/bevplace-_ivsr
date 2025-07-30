
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights
import math
import einops

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        N, C = x.shape[:2]
        x_flatten = x.view(N, C, -1)
        
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class REIN(nn.Module):
    def __init__(self):
        super(REIN, self).__init__()
        self.rem = REM()
        self.pooling = NetVLAD()
        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim * 64
    
    def forward(self, x):
        out1, global_desc_init = self.rem(x)
        global_desc = self.pooling(out1)  # Use NetVLAD for final global descriptor
        return out1, global_desc

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.d_inner = int(expand * dim)
        
        # Projects to higher dimension for Mamba processing
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding='same',
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_conv + 1)
        self.dt_proj = nn.Linear(self.d_inner, self.d_state)
        
        # Initialize A, B, C for state space model with proper shapes
        self.A = nn.Parameter(torch.randn(self.d_state) / self.d_state**0.5)  # [d_state]
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state) / self.d_state**0.5)  # [d_inner, d_state]
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state) / self.d_state**0.5)  # [d_inner, d_state]
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim)
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # Input projection and splitting
        x_and_res = self.in_proj(x)  # [batch, seq_len, 2 * d_inner]
        x_copy, res = x_and_res.chunk(2, dim=-1)
        
        # Convolution branch
        conv_state = self.conv1d(x_copy.permute(0, 2, 1)).permute(0, 2, 1)
        
        # SSM branch
        x_ssm = self.x_proj(conv_state)  # [batch, seq_len, d_state + d_conv + 1]
        delta, beta, gamma = torch.split(x_ssm, [self.d_state, self.d_conv, 1], dim=-1)
        delta = F.softplus(self.dt_proj(conv_state))
        
        # Discretize continuous-time SSM into discrete-time
        # Proper broadcasting for A matrix
        dA = torch.exp(-delta * F.softplus(self.A.view(1, 1, -1)))  # [batch, seq_len, d_state]
        dB = F.softplus(self.B)  # [d_inner, d_state]
        dC = self.C  # [d_inner, d_state]
        
        # Run discretized SSM with proper dimension handling
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        outs = []
        
        # Reshape A, B, C matrices for broadcasting
        dA = dA.view(batch, seq_len, 1, self.d_state)  # [B, seq_len, 1, d_state]
        dB = dB.view(1, 1, self.d_inner, self.d_state)  # [1, 1, d_inner, d_state]
        dC = dC.view(1, 1, self.d_inner, self.d_state)  # [1, 1, d_inner, d_state]
        
        # Reshape input for broadcasting
        x_copy = x_copy.view(batch, seq_len, self.d_inner, 1)  # [B, seq_len, d_inner, 1]
        
        for t in range(seq_len):
            # Update hidden state with proper broadcasting
            h = h.view(batch, 1, self.d_inner, self.d_state) * dA[:, t:t+1]  # [B, 1, d_inner, d_state]
            h = h + x_copy[:, t:t+1] * dB  # [B, 1, d_inner, d_state]
            h = h.view(batch, self.d_inner, self.d_state)  # [B, d_inner, d_state]
            
            # Compute output with proper broadcasting
            y = (h.unsqueeze(1) * dC).sum(dim=-1)  # [B, 1, d_inner]
            outs.append(y)
        
        out = torch.cat(outs, dim=1)  # [B, seq_len, d_inner]
        out = out * F.silu(gamma) + res
        
        return self.out_proj(out)

class VisionMamba(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, d_state=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state=d_state)
            for _ in range(depth)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        B, C, H, W = x.shape
        
        # Ensure input size matches expected size
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
            
        return x  # [B, num_patches + 1, embed_dim]

class CNNProcessor(nn.Module):
    def __init__(self, num_patches, input_channels=768, output_size=26):
        super(CNNProcessor, self).__init__()
        self.output_size = output_size
        self.num_patches = num_patches
        
        # Calculate strides dynamically based on num_patches
        height = int((num_patches + 1) ** 0.5)  # Approximate height for square patch grid
        stride = max(1, (height + output_size - 1) // output_size)  # Dynamic stride for both dimensions
        
        # CNN architecture with square kernels - now accepting input_channels
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=5, stride=stride, padding=2)  # Increased channels for first layer
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Final output 128 channels
        self.bn4 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        # Input shape: [batch_size, 1, num_patches + 1, 768]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 32, ~output_size, ~output_size]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, ~output_size, ~output_size]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, ~output_size, ~output_size]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 128, ~output_size, ~output_size]
        
        # Pad or crop to exactly [batch_size, 128, output_size, output_size]
        current_h, current_w = x.shape[2], x.shape[3]
        if current_h != self.output_size or current_w != self.output_size:
            # Calculate padding or cropping
            pad_h = max(0, self.output_size - current_h)
            pad_w = max(0, self.output_size - current_w)
            crop_h = max(0, current_h - self.output_size)
            crop_w = max(0, current_w - self.output_size)
            
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')  # Use replicate padding
            if crop_h > 0 or crop_w > 0:
                x = x[:, :, :self.output_size, :self.output_size]  # Crop to output_size
            
        # Output shape: [batch_size, 128, output_size, output_size]
        return x

class REM(nn.Module):
    def __init__(self, image_size=224, patch_size=16, from_scratch=True, rotations=8):
        super(REM, self).__init__()
        
        # Use Vision Mamba + CNNProcessor
        self.mamba = VisionMamba(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=768,
            depth=12,
            d_state=16
        )
        self.cnn = CNNProcessor(num_patches=(image_size // patch_size) ** 2, input_channels=768, output_size=26)  # 768 is embedding dim

        # Rotations
        self.angles = -torch.arange(0, 359.00001, 360.0/rotations)/180*torch.pi

    def forward(self, x):
        equ_features = []
        batch_size = x.size(0)

        for i in range(len(self.angles)):
            # Input warp grids
            aff = torch.zeros(batch_size, 2, 3).cuda()
            aff[:, 0, 0] = torch.cos(-self.angles[i])
            aff[:, 0, 1] = torch.sin(-self.angles[i])
            aff[:, 1, 0] = -torch.sin(-self.angles[i])
            aff[:, 1, 1] = torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, list(x.size()), align_corners=True).type(x.type())

            # Input warp
            warped_im = F.grid_sample(x, grid, align_corners=True, mode='bicubic')

            # Mamba backbone feature
            mamba_out = self.mamba(warped_im)  # [B, num_patches+1, embed_dim]
            # Remove class token, reshape for CNN
            mamba_out = mamba_out[:, 1:, :]  # [B, num_patches, embed_dim]
            B, num_patches, hidden_dim = mamba_out.shape
            h = w = int(num_patches ** 0.5)
            mamba_out = mamba_out.transpose(1, 2).reshape(B, hidden_dim, h, w)  # [B, hidden_dim, h, w]

            # CNN backbone feature
            out = self.cnn(mamba_out)  # [B, 128, 26, 26]

            # Output feature warp grids
            if i == 0:
                im1_init_size = out.size()

            aff = torch.zeros(batch_size, 2, 3).cuda()
            aff[:, 0, 0] = torch.cos(self.angles[i])
            aff[:, 0, 1] = torch.sin(self.angles[i])
            aff[:, 1, 0] = -torch.sin(self.angles[i])
            aff[:, 1, 1] = torch.cos(self.angles[i])
            grid = F.affine_grid(aff, list(im1_init_size), align_corners=True).type(x.type())

            # Output feature warp
            out = F.grid_sample(out, grid, align_corners=True, mode='bicubic')
            equ_features.append(out.unsqueeze(-1))

        equ_features = torch.cat(equ_features, dim=-1)  # [B, C, H, W, R]
        B, C, H, W, R = equ_features.shape
        equ_features = torch.max(equ_features, dim=-1, keepdim=False)[0]  # [B, 128, 26, 26]

        aff = torch.zeros(batch_size, 2, 3).cuda()
        aff[:, 0, 0] = 1
        aff[:, 0, 1] = 0
        aff[:, 1, 0] = 0
        aff[:, 1, 1] = 1

        # Upsample for NetVLAD
        B, C, H_orig, W_orig = x.size()
        grid = F.affine_grid(aff, [B, 128, H_orig, W_orig], align_corners=True)
        out1 = F.grid_sample(equ_features, grid, align_corners=True, mode='bicubic')
        out1 = F.normalize(out1, dim=1)

        # Create global descriptor by averaging over spatial dimensions
        global_desc = F.adaptive_avg_pool2d(out1, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 128]
        global_desc = F.normalize(global_desc, p=2, dim=1)  # L2 normalize

        return out1, global_desc
