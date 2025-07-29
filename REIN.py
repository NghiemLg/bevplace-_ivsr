
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights

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

class VisionTransformerBase(nn.Module):
    def __init__(self, image_size=224, patch_size=16, pretrained=False):
        super(VisionTransformerBase, self).__init__()
        # Load ViT-B/16 with weights=None for training from scratch
        weights = None if not pretrained else ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=weights)
        
        # Extract and reconfigure components
        self.conv_proj = self.vit.conv_proj  # Patch embedding convolution
        self.encoder = self.vit.encoder
        self.class_token = nn.Parameter(self.vit.class_token.data.clone())
        
        # Initialize dynamic positional embedding
        self.hidden_dim = self.vit.hidden_dim
        self.patch_size = patch_size
        self.num_channels = 3  # Assuming RGB input
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, self.hidden_dim, device='cuda'))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # Input shape: [batch_size, C, H, W]
        batch_size, C, H, W = x.shape
        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {C}")

        # Always resize input to 224x224 for ViT
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Forward through ViT components
        # Patch embedding
        x = self.vit.conv_proj(x)  # [B, hidden_dim, grid, grid]
        
        # Flatten
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, hidden_dim]
        
        # Add class token
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embedding
        
        # Transformer encoder
        x = self.vit.encoder(x)  # [B, n_patches + 1, hidden_dim]
        
        return x

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
        
        # Replace resnet34 with VisionTransformerBase + CNNProcessor
        self.vit = VisionTransformerBase(image_size=image_size, patch_size=patch_size, pretrained=not from_scratch)
        self.cnn = CNNProcessor(num_patches=(image_size // patch_size) ** 2, input_channels=768, output_size=26)  # 768 is ViT's hidden_dim

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

            # ViT backbone feature
            vit_out = self.vit(warped_im)  # [B, num_patches+1, hidden_dim]
            # Remove class token, reshape for CNN
            vit_out = vit_out[:, 1:, :]  # [B, num_patches, hidden_dim]
            B, num_patches, hidden_dim = vit_out.shape
            h = w = int(num_patches ** 0.5)
            vit_out = vit_out.transpose(1, 2).reshape(B, hidden_dim, h, w)  # [B, hidden_dim, h, w]

            # CNN backbone feature
            out = self.cnn(vit_out)  # [B, 128, 26, 26]

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
