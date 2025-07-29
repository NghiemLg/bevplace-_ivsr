
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

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
        dots = dots[::-1, :] # sort, descending

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
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
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
        self.global_feat_dim = self.local_feat_dim*64
    
    def forward(self, x):
        out1, local_feats = self.rem(x)
        global_desc = self.pooling(out1)
        return out1, local_feats, global_desc
    def __init__(self):
        super(CNNProcessor, self).__init__()
        
        # CNN architecture with smaller stride for better spatial relationship
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 20), stride=(4, 20), padding=(2, 10))  # [B, 32, 49, 38]
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1)  # [B, 64, 25, 19]
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # [B, 128, 25, 19]
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))  # [B, 128, 25, 10]
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))  # [B, 128, 25, 5]
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 5), padding=(1, 0))  # [B, 128, 25, 1]
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=1, padding=0)  # [B, 128, 25, 1]
        self.bn7 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        # Input shape: [batch_size, 1, 197, 768]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 32, 49, 38]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, 25, 19]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, 25, 19]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 128, 25, 10]
        x = F.relu(self.bn5(self.conv5(x)))  # [B, 128, 25, 5]
        x = F.relu(self.bn6(self.conv6(x)))  # [B, 128, 25, 1]
        x = F.relu(self.bn7(self.conv7(x)))  # [B, 128, 25, 1]
        
        # Pad to [B, 128, 26, 26]
        x = F.pad(x, (0, 25, 0, 1))  # Pad width from 1 to 26, height from 25 to 26
        
        # Flatten to [batch_size, 128*26*26]
        x = x.view(x.size(0), -1)  # [B, 128*26*26]
        
        return x


class VisionTransformerBase(nn.Module):
    def __init__(self, image_size=224, patch_size=16, pretrained=False):
        super(VisionTransformerBase, self).__init__()
        # Load the ViT-B/16 model without pretrained weights
        vit = models.vit_b_16(pretrained=pretrained)
        
        # Extract the transformer backbone without final linear layer
        self.conv_norm = vit.conv_proj
        self.encoder = vit.encoder
        self.patch_embed = vit.patch_embed
        
        # Keep class token embedding
        self.class_token = nn.Parameter(vit.class_token.data.clone())
        
        # Store configuration
        self.hidden_dim = vit.hidden_dim
        self.image_size = image_size  # Dynamic image size (H=W)
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        # Input shape: [batch_size, C, H, W]
        # Validate input size
        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(f"Expected input size [batch_size, 3, {self.image_size}, {self.image_size}], got {x.shape}")
        
        x = self.conv_norm(x)  # [B, hidden_dim, H', W']
        x = self.patch_embed(x)  # [B, num_patches, hidden_dim]
        
        # Add class token
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([class_token, x], dim=1)  # [B, num_patches + 1, hidden_dim]
        
        # Process through transformer encoder
        x = self.encoder(x)  # [B, num_patches + 1, hidden_dim]
        
        # Reshape to [batch_size, 1, num_patches + 1, hidden_dim]
        x = x.unsqueeze(1)  # [B, 1, num_patches + 1, hidden_dim]
        
        return x

class CNNProcessor(nn.Module):
    def __init__(self, num_patches, output_size=26):
        super(CNNProcessor, self).__init__()
        self.output_size = output_size
        self.num_patches = num_patches
        
        # Calculate strides dynamically based on num_patches
        height = int((num_patches + 1) ** 0.5)  # Approximate height for square patch grid
        stride_h = max(1, (height + output_size - 1) // output_size)  # Dynamic stride for height
        stride_w = max(1, (768 + output_size - 1) // output_size)  # Dynamic stride for width
        
        # CNN architecture with dynamic strides
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 20), stride=(stride_h, stride_w), padding=(2, 10))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=(1, 1), padding=1)
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
                x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad with zeros
            if crop_h > 0 or crop_w > 0:
                x = x[:, :, :self.output_size, :self.output_size]  # Crop to output_size
            
        # Output shape: [batch_size, 128, output_size, output_size]
        return x

class REM(nn.Module):
    def __init__(self, image_size=224, patch_size=16, from_scratch=True, rotations=8):
        super(REM, self).__init__()
        
        # Replace resnet34 with VisionTransformerBase + CNNProcessor
        self.vit = VisionTransformerBase(image_size=image_size, patch_size=patch_size, pretrained=not from_scratch)
        self.cnn = CNNProcessor(num_patches=(image_size // patch_size) ** 2, output_size=26)
        self.encoder = nn.Sequential(self.vit, self.cnn)

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
                                    
            # CNN backbone feature (VisionTransformerBase + CNNProcessor)
            out = self.encoder(warped_im)  # [B, 128, 26, 26]

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
        
        # Upsample for keypoints
        grid = F.affine_grid(aff, [B, 128, H_orig, W_orig], align_corners=True)
        out2 = F.grid_sample(equ_features, grid, align_corners=True, mode='bicubic')
        out2 = F.normalize(out2, dim=1)
        
        return out1, out2