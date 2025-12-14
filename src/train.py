"""
CSIRO - Image2Biomass Competition
Training Script with Multi-Task Learning
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import warnings
import utils as ut
import data_loader as dl
from config import Config
from model import MultiTaskBiomassModel
from loss import WeightedMSELoss

warnings.filterwarnings('ignore')

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion_biomass, criterion_aux_reg, criterion_aux_cls, optimizer):
    model.train()
    total_loss = 0
    biomass_loss_total = 0
    aux_loss_total = 0
    
    for images, biomass_targets, auxiliary_targets in loader:
        images = images.to(Config.DEVICE)
        biomass_targets = biomass_targets.to(Config.DEVICE)
        
        # Apply mixup if enabled
        if Config.USE_MIXUP and np.random.random() < 0.5:
            images, targets_a, targets_b, lam = dl.mixup_data(images, biomass_targets, Config.MIXUP_ALPHA)
        else:
            targets_a = biomass_targets
            targets_b = None
            lam = 1.0
        
        optimizer.zero_grad()
        
        if Config.USE_MULTITASK:
            outputs = model(images, return_auxiliary=True)
            
            # Biomass loss with mixup
            if targets_b is not None:
                biomass_loss = mixup_criterion(criterion_biomass, outputs['biomass'], targets_a, targets_b, lam)
            else:
                biomass_loss = criterion_biomass(outputs['biomass'], targets_a)
            
            # Auxiliary losses (no mixup for auxiliary tasks)
            aux_loss = 0
            aux_loss += criterion_aux_reg(outputs['ndvi'], auxiliary_targets['ndvi'].to(Config.DEVICE))
            aux_loss += criterion_aux_reg(outputs['height'], auxiliary_targets['height'].to(Config.DEVICE))
            aux_loss += criterion_aux_cls(outputs['state'], auxiliary_targets['state'].to(Config.DEVICE).squeeze())
            aux_loss += criterion_aux_cls(outputs['species'], auxiliary_targets['species'].to(Config.DEVICE).squeeze())
            aux_loss = aux_loss / 4
            
            loss = biomass_loss + Config.AUXILIARY_WEIGHT * aux_loss
            biomass_loss_total += biomass_loss.item()
            aux_loss_total += aux_loss.item()
        else:
            outputs = model(images, return_auxiliary=False)
            
            if targets_b is not None:
                loss = mixup_criterion(criterion_biomass, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion_biomass(outputs, targets_a)
            biomass_loss_total += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    n = len(loader)
    return total_loss / n, biomass_loss_total / n, aux_loss_total / n


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, biomass_targets, _ in loader:
            images = images.to(Config.DEVICE)
            biomass_targets = biomass_targets.to(Config.DEVICE)
            
            outputs = model(images, return_auxiliary=False)
            loss = criterion(outputs, biomass_targets)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(biomass_targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    
    return total_loss / len(loader), rmse, mae


def train_model():
    """Main training function"""
    
    ut.set_seed(Config.SEED)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("CSIRO - Image2Biomass Training")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.BACKBONE_MODEL}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Multi-task learning: {Config.USE_MULTITASK}")
    print(f"Mixup: {Config.USE_MIXUP}" + (f" (alpha={Config.MIXUP_ALPHA})" if Config.USE_MIXUP else ""))
    
    print("\n" + "="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    train_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'train.csv'))
    print(f"Raw train shape: {train_df.shape}")
    
    train_wide = dl.pivot_train_data(train_df)
    print(f"Wide train shape: {train_wide.shape}")
    
    gkf = GroupKFold(n_splits=Config.NUM_FOLDS)
    groups = train_wide['image_id'].values
    
    # Determine which folds to train based on configuration
    folds_to_train = range(Config.NUM_FOLDS) if Config.TRAIN_FOLD == -1 else [Config.TRAIN_FOLD]
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_wide, groups=groups)):
        if fold_idx not in folds_to_train:
            continue
        
        print("\n" + "="*60)
        print(f"Training Fold {fold_idx + 1}/{Config.NUM_FOLDS}")
        print("="*60)
        
        train_data_raw = train_wide.iloc[train_idx].reset_index(drop=True)
        val_data_raw = train_wide.iloc[val_idx].reset_index(drop=True)
        

        # Fit and Transform
        train_data, val_data, scalers, label_encoders = dl.fit_and_transform_fold_data(
            train_data_raw, val_data_raw
        )

        # Create the preprocessing info dictionary here
        preprocessing_info = {
            'scalers': scalers,
            'label_encoders': label_encoders,
            'num_states': len(label_encoders['State'].classes_) + 1,
            'num_species': len(label_encoders['Species'].classes_) + 1,
            'target_names': Config.TARGET_NAMES
        }
        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

        with open(os.path.join(Config.OUTPUT_DIR, f'preprocessing_info_fold{fold_idx}.pkl'), 'wb') as f:
            pickle.dump(preprocessing_info, f)
        
        # Create datasets
        train_dataset = dl.BiomassDataset(train_data, Config.DATA_DIR, dl.get_transforms(True))
        val_dataset = dl.BiomassDataset(val_data, Config.DATA_DIR, dl.get_transforms(False))
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                                 shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                               shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        # Initialize model
        model = MultiTaskBiomassModel(
            Config.BACKBONE_MODEL,
            num_biomass_targets=len(Config.TARGET_NAMES),
            num_states=preprocessing_info['num_states'],
            num_species=preprocessing_info['num_species'],
            use_multitask=Config.USE_MULTITASK
        ).to(Config.DEVICE)
        
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion_biomass = WeightedMSELoss(Config.TARGET_WEIGHTS, Config.DEVICE) 
        criterion_aux_reg = nn.MSELoss()
        criterion_aux_cls = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print("\nStarting training...\n")
        
        for epoch in range(Config.NUM_EPOCHS):
            train_loss, biomass_loss, aux_loss = train_epoch(
                model, train_loader, criterion_biomass, criterion_aux_reg,
                criterion_aux_cls, optimizer
            )
            val_loss, val_rmse, val_mae = validate(model, val_loader, criterion_biomass)
            
            train_losses.append(biomass_loss) # only report the biomass loss for fair comparison with the val_losses
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1:02d}/{Config.NUM_EPOCHS}")
            if Config.USE_MULTITASK:
                print(f"  Train - Total: {train_loss:.4f} | Biomass: {biomass_loss:.4f} | Aux: {aux_loss:.4f}")
            else:
                print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save model
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'model_name': Config.BACKBONE_MODEL,
                        'num_biomass_targets': len(Config.TARGET_NAMES),
                        'num_states': preprocessing_info['num_states'],
                        'num_species': preprocessing_info['num_species'],
                        'use_multitask': Config.USE_MULTITASK
                    },
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae
                }
                torch.save(checkpoint, os.path.join(Config.OUTPUT_DIR, f'best_model_fold{fold_idx}.pth'))
                print("Best model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        ut.plot(os.path.join(Config.OUTPUT_DIR, f'train_graph_fold{fold_idx}.png'), train_losses, val_losses)
        print(f"\nBest validation loss for fold {fold_idx}: {best_val_loss:.4f}")
        print(f"Model saved to: {Config.OUTPUT_DIR}/best_model{fold_idx}.pth")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    train_model()
