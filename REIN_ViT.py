import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from PIL import Image
import math
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from RANSAC import rigidRansac


class REIN_ViT(nn.Module):
    def __init__(self, input_dim=768, output_dim=8192, rotations=8):
        super(REIN_ViT, self).__init__()
        
        # Load ViT architecture (not pretrained)
        self.vit = models.vit_b_16(weights=None)
        
        # Cắt bỏ lớp classifier
        self.vit.heads = nn.Identity()
        
        # Lớp linear để biến đổi class embedding với activation và normalization
        self.class_embedding_projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Layer normalization cho tokens
        self.token_norm = nn.LayerNorm(input_dim)
        
        # Convolution layer để xử lý local features
        self.local_conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layer để tăng resolution cho local features
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=4, padding=0),  # 14x14 -> 56x56
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=4, padding=0),  # 56x56 -> 224x224
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True)
        )
        
        # Projection layer để giảm dimensions từ 768 -> 128 (tương thích với model cũ)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Rotations
        self.angles = -torch.arange(0, 359.00001, 360.0/rotations)/180*torch.pi
        
        # Các thuộc tính cần thiết cho NetVLAD
        self.local_feat_dim = 128  # Đã được project từ 768 -> 128
        self.global_feat_dim = output_dim  # 8192
        
        # ViT preprocessing transforms
        self.vit_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization for RGB
        ])
        
        # Original BEV image size
        self.original_size = 201
        self.vit_size = 224
    
    def map_coordinates(self, kp_coords, original_size=201, target_size=224):
        """Map keypoint coordinates from original image space to feature space"""
        # kp_coords: [N, 2] with (x, y) coordinates in original image space
        # Return: [N, 2] with (x, y) coordinates in feature space
        
        # Scale factor
        scale_x = target_size / original_size
        scale_y = target_size / original_size
        
        # Apply scaling
        mapped_coords = kp_coords * np.array([scale_x, scale_y])
        
        # Clip to valid range
        mapped_coords[:, 0] = np.clip(mapped_coords[:, 0], 0, target_size - 1)
        mapped_coords[:, 1] = np.clip(mapped_coords[:, 1], 0, target_size - 1)
        
        return mapped_coords.astype(np.int32)
    
    def preprocess_bev_image(self, x):
        """Preprocess BEV images for ViT input"""
        # x có shape [B, 3, 201, 201] với range [0, 1]
        batch_size = x.size(0)
        processed_images = []
        
        for i in range(batch_size):
            # Convert to PIL Image
            img = x[i].permute(1, 2, 0).cpu().numpy()  # [201, 201, 3]
            img = (img * 255).astype(np.uint8)  # [0, 255]
            pil_img = Image.fromarray(img[:, :, 0])  # Take first channel (grayscale)
            
            # Convert grayscale to RGB for ViT
            pil_img_rgb = pil_img.convert('RGB')
            
            # Apply ViT transforms
            processed_img = self.vit_transforms(pil_img_rgb)  # [3, 224, 224] with grayscale normalization
            processed_images.append(processed_img)
        
        return torch.stack(processed_images).to(x.device)
    
    def extract_features_at_keypoints(self, local_feats, keypoints, original_size=201):
        """Extract local features at specific keypoint coordinates"""
        # local_feats: [B, 128, 224, 224] - upsampled features
        # keypoints: list of cv2.KeyPoint objects from original image
        # Return: [N, 128] feature descriptors
        
        if len(keypoints) == 0:
            return np.array([])
        
        # Extract coordinates
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Map coordinates to feature space
        mapped_coords = self.map_coordinates(kp_coords, original_size, 224)
        
        # Extract features
        features = []
        for coord in mapped_coords:
            x, y = coord[0], coord[1]
            # Note: local_feats has shape [B, C, H, W], so we need [y, x] for indexing
            feat = local_feats[0, :, y, x].cpu().numpy()  # [128]
            features.append(feat)
        
        return np.array(features)
    
    @staticmethod
    def pose_estimation_with_vit_features(query_im, db_im, query_kps, db_kps, local_feats_query, local_feats_db, model):
        """Pose estimation using ViT features with proper coordinate mapping"""
        # query_im, db_im: original images (201x201)
        # query_kps, db_kps: keypoints detected on original images
        # local_feats_query, local_feats_db: ViT local features [B, 128, 224, 224]
        # model: REIN_ViT model instance
        
        # Extract features at keypoints with coordinate mapping
        query_des = model.extract_features_at_keypoints(local_feats_query, query_kps, original_size=201)
        db_des = model.extract_features_at_keypoints(local_feats_db, db_kps, original_size=201)
        
        if len(query_des) == 0 or len(db_des) == 0:
            return None, None, 0
        
        # Feature matching
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(query_des, db_des, k=2)
        
        # Apply ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2 and match[0].distance < 0.7 * match[1].distance:
                good_matches.append(match[0])
        
        if len(good_matches) < 4:
            return None, None, len(good_matches)
        
        # Extract matched points
        points1 = np.float32([query_kps[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([db_kps[m.trainIdx].pt for m in good_matches])
        
        # RANSAC pose estimation
        im_side = query_im.shape[0]  # Should be 201
        H, mask, max_csc_num = rigidRansac(
            (np.array([[im_side//2, im_side//2]]) - points1) * 0.4,
            (np.array([[im_side//2, im_side//2]]) - points2) * 0.4
        )
        
        return H, mask, len(good_matches)
    
    def get_vit_full_tokens(self, x):
        """Lấy tất cả tokens từ ViT"""
        x = self.vit._process_input(x)  # [B, 196, 768]
        n = x.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder(x)  # [B, 197, 768]
        return x
    
    def forward(self, x):
        # Preprocess BEV images for ViT
        x = self.preprocess_bev_image(x)
        
        batch_size = x.size(0)
        equ_features = []
        
        # Rotation loop tương tự như trong REM
        for i in range(len(self.angles)):
            # Input warp grids
            aff = torch.zeros(batch_size, 2, 3).cuda()
            aff[:, 0, 0] = torch.cos(-self.angles[i])
            aff[:, 0, 1] = torch.sin(-self.angles[i])
            aff[:, 1, 0] = -torch.sin(-self.angles[i])
            aff[:, 1, 1] = torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, torch.Size(x.size()), align_corners=True).type(x.type())
            
            # Input warp
            warped_im = F.grid_sample(x, grid, align_corners=True, mode='bicubic')
            
            # ViT backbone feature
            tokens = self.get_vit_full_tokens(warped_im)  # [B, 197, 768]
            
            # Normalize tokens
            tokens = self.token_norm(tokens)
            
            # Output feature warp grids (để đưa về orientation gốc)
            if i == 0:
                tokens_init_size = tokens.size()
            
            aff = torch.zeros(batch_size, 2, 3).cuda()
            aff[:, 0, 0] = torch.cos(self.angles[i])
            aff[:, 0, 1] = torch.sin(self.angles[i])
            aff[:, 1, 0] = -torch.sin(self.angles[i])
            aff[:, 1, 1] = torch.cos(self.angles[i])
            
            # Reshape tokens để có thể apply grid_sample
            # tokens: [B, 197, 768] -> [B, 768, 14, 14] (giả sử 14x14 patches)
            B, num_tokens, dim = tokens.shape
            H = W = int(np.sqrt(num_tokens - 1))  # Bỏ qua class token
            patch_tokens = tokens[:, 1:, :]  # [B, 196, 768]
            tokens_reshaped = patch_tokens.view(B, H, W, dim).permute(0, 3, 1, 2)  # [B, 768, 14, 14]
            
            grid = F.affine_grid(aff, torch.Size(tokens_reshaped.size()), align_corners=True).type(x.type())
            
            # Output feature warp
            tokens_warped = F.grid_sample(tokens_reshaped, grid, align_corners=True, mode='bicubic')
            
            # Reshape lại về dạng tokens
            tokens_warped = tokens_warped.permute(0, 2, 3, 1).view(B, H*W, dim)  # [B, 196, 768]
            class_token = tokens[:, 0:1, :]  # Giữ nguyên class token
            tokens_final = torch.cat((class_token, tokens_warped), dim=1)  # [B, 197, 768]
            
            equ_features.append(tokens_final.unsqueeze(-1))
        
        # Concatenate tất cả rotations
        equ_features = torch.cat(equ_features, axis=-1)  # [B, 197, 768, R]
        
        # Max pooling along rotations
        B, num_tokens, dim, R = equ_features.shape
        equ_features = torch.max(equ_features, dim=-1, keepdim=False)[0]  # [B, 197, 768]
        
        # Lấy class token và patch tokens
        class_token = equ_features[:, 0, :]  # [B, 768]
        patch_tokens = equ_features[:, 1:, :]  # [B, 196, 768]
        
        # Project class token qua linear layer với activation
        global_desc = self.class_embedding_projector(class_token)  # [B, 8192]
        
        # Normalize global descriptor
        global_desc = F.normalize(global_desc, p=2, dim=1)
        
        # Tạo local features từ patch tokens
        H = W = int(np.sqrt(patch_tokens.size(1)))  # 14x14
        local_feats = patch_tokens.view(B, H, W, dim).permute(0, 3, 1, 2)  # [B, 768, 14, 14]
        
        # Xử lý local features qua convolution layers
        local_feats = self.local_conv(local_feats)
        
        # Upsample để tăng resolution (14x14 -> 224x224)
        local_feats_high_res = self.upsample(local_feats)  # [B, 768, 224, 224]
        
        # Project để giảm dimensions (768 -> 128) cho tương thích với model cũ
        local_feats_128 = self.feature_projection(local_feats_high_res)  # [B, 128, 224, 224]
        
        # Normalize local features
        local_feats = F.normalize(local_feats_128, dim=1)
        
        return local_feats, local_feats, global_desc  # Trả về 3 outputs như REIN gốc