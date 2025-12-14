"""
Model Architecture
"""

import torch.nn as nn
from config import Config
import torch.hub
from torchvision import models

class MultiTaskBiomassModel(nn.Module):
    """
    Vision-only model with auxiliary tasks
    
    This model saves all weights (including backbone) in the checkpoint,
    so it can be loaded in Kaggle's offline mode without downloading pretrained weights.
    """
    
    def __init__(self, model_name, num_biomass_targets, num_states, num_species, 
                 use_multitask, pretrained=True):
        """
        Args:
            model_name: Backbone architecture name
            num_biomass_targets: Number of biomass prediction targets (5)
            num_states: Number of state classes
            num_species: Number of species classes
            use_multitask: Whether to use multi-task learning
            pretrained: Use pretrained weights (True for training, False for inference)
        """
        super(MultiTaskBiomassModel, self).__init__()
        
        self.use_multitask = use_multitask
        self.model_name = model_name
        
        # Backbone initialization
        # During training: pretrained=True (download ImageNet weights)
        # During inference: pretrained=False (weights loaded from checkpoint)
        if model_name == 'efficientnet_b0':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b3':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'resnet50':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'convnext_small':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            feat_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'convnext_base':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.convnext_base(weights=weights)
            feat_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'dinov2_small':
            # ViT-Small/14, 384 features
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 
                'dinov2_vits14', 
                pretrained=pretrained
            ).to(Config.DEVICE)
            feat_dim = 384
        elif model_name == 'dinov2_base':
            # ViT-Base/14, 768 features
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 
                'dinov2_vitb14', 
                pretrained=pretrained
            ).to(Config.DEVICE)
            feat_dim = 768
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Feature extractor - shared across all tasks
        self.feature_extractor = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Main task. Biomass regression head
        self.biomass_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_biomass_targets)
        )
        
        # Auxiliary heads, for regularization during training
        if use_multitask:
            # NDVI regression head
            self.ndvi_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            # Height regression head
            self.height_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            # State classification head
            self.state_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_states)
            )
            
            # Species classification head
            self.species_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_species)
            )

        self.freeze_backbone()
    
    def forward(self, images, return_auxiliary=False):
        # Extract image features
        img_features = self.backbone(images)
        
        if self.model_name.startswith('dinov2'):
            # DINOv2 (ViT) output is typically 3D: (B, L, D). 
            # Only extract the CLS token (idx 0) if the tensor is 3D.
            if img_features.dim() == 3:
                img_features = img_features[:, 0, :]
            # If img_features.dim() is 2 the CLS token is already extracted, so skip this
            
        elif img_features.dim() == 4:
            # CNN backbones output a 4D tensor (B, C, H, W).
            # Apply Global Average Pooling (GAP) to collapse H and W. (fix for ConvNeXt)
            img_features = img_features.mean(dim=[2, 3])

        # Shared feature representation
        shared_features = self.feature_extractor(img_features)
        
        # Main biomass prediction
        biomass_pred = self.biomass_head(shared_features)
        
        # Return auxiliary outputs if requested (during training)
        if return_auxiliary and self.use_multitask:
            return {
                'biomass': biomass_pred,
                'ndvi': self.ndvi_head(shared_features),
                'height': self.height_head(shared_features),
                'state': self.state_head(shared_features),
                'species': self.species_head(shared_features)
            }
        
        # Return only biomass predictions (during inference)
        return biomass_pred
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_batch_norms(self):
        """Freeze batch normalization layers"""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

