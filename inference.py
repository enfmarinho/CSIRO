"""
CSIRO - Image2Biomass Competition
Inference Script for Generating Submissions

This script supports single-fold inference and multi-fold ensembling (averaging
predictions from multiple trained models).

This script also needs the config and the model scripts to work
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import warnings

from config import Config as TrainConfig
from model import MultiTaskBiomassModel

def extract_image_id(sample_id):
    """Extract image ID from sample_id"""
    return sample_id.split('__')[0]

warnings.filterwarnings('ignore')

# Append inference specific configuration
class Config(TrainConfig):
    DATA_DIR = '/kaggle/input/csiro-biomass'
    MODEL_DIR = '/kaggle/input/csiro/pytorch/default/1' 
    
    FOLDS_TO_INFER = [0, 1, 2, 3, 4] 

    # Inference settings
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    
    # Test-Time Augmentation (TTA)
    USE_TTA = True 
    TTA_ROTATIONS = True
    
    OUTPUT_PATH = 'submission.csv'

class BiomassTestDataset(Dataset):
    """Dataset for test/inference"""
    
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, row['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Load a placeholder gray image on error
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='gray') 
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and the unique image_id
        return image, row['image_id']


def get_transforms_tta():
    """Base test transform for TTA"""
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_models_for_ensemble(folds):
    """
    Loads trained models and associated preprocessing info for the specified folds.
    
    Returns: A list of (model, preprocessing_info) tuples and target_names.
    """
    
    print("="*60)
    print(f"Loading {len(folds)} model(s) for ensemble...")
    print("="*60)
    
    ensemble_info = []
    target_names = None
    
    for fold_idx in folds:
        model_path = os.path.join(Config.MODEL_DIR, f'best_model_fold{fold_idx}.pth')
        preprocessing_path = os.path.join(Config.MODEL_DIR, f'preprocessing_info_fold{fold_idx}.pkl')
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model for fold {fold_idx} not found at {model_path}. Skipping.")
            continue
        if not os.path.exists(preprocessing_path):
             print(f"WARNING: Preprocessing info for fold {fold_idx} not found at {preprocessing_path}. Skipping.")
             continue

        try:
            # Load preprocessing info to get target names and num_classes
            with open(preprocessing_path, 'rb') as f:
                preprocessing_info = pickle.load(f)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=Config.DEVICE)
            
            model_config = checkpoint['config']
            
            model = MultiTaskBiomassModel(
                model_name=model_config['model_name'],
                num_biomass_targets=model_config['num_biomass_targets'],
                num_states=model_config['num_states'],
                num_species=model_config['num_species'],
                use_multitask=model_config['use_multitask'],
                pretrained=False
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(Config.DEVICE)
            model.eval()
            
            print(f"Fold {fold_idx} Model Loaded: {model_config['model_name']} | Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

            ensemble_info.append((model, preprocessing_info))
            
            # Set target names from the first loaded fold
            if target_names is None:
                target_names = preprocessing_info['target_names'] 

        except Exception as e:
            print(f"ERROR loading fold {fold_idx}: {e}. Skipping.")
            continue


    if not ensemble_info:
        raise RuntimeError("No models were successfully loaded for inference.")
        
    return ensemble_info, target_names


def predict(ensemble_info, test_loader):
    """Make predictions on test set using the ensemble of models and TTA (if enabled)."""
    
    num_models = len(ensemble_info)
    print("\n" + "="*60)
    print(f"Making predictions with {num_models} model(s)...")
    if Config.USE_TTA:
        tta_type = "Flips Only" if not Config.TTA_ROTATIONS else "Flips + Rotations"
        print(f"Using Test-Time Augmentation (TTA): {tta_type}")
    print("="*60)
    
    all_predictions = [] # Stores predictions for all images *for all models*
    image_ids = []
    
    with torch.no_grad():
        for images, img_ids in test_loader:
            # Per-Model Prediction Loop
            images = images.to(Config.DEVICE)
            
            batch_ensemble_preds = [] # Stores TTA-averaged prediction for one batch
            
            for model, _ in ensemble_info:
                tta_predictions = []
                
                # Base prediction (No TTA)
                tta_predictions.append(model(images)) 

                if Config.USE_TTA:
                    # Horizontal flip
                    tta_predictions.append(model(torch.flip(images, dims=[3])))
                    
                    # Vertical flip
                    tta_predictions.append(model(torch.flip(images, dims=[2])))
                    
                    if Config.TTA_ROTATIONS:
                        # 90 degree rotation
                        tta_predictions.append(model(torch.rot90(images, k=1, dims=[2, 3])))
                        # 270 degree rotation
                        tta_predictions.append(model(torch.rot90(images, k=3, dims=[2, 3])))
                    
                    # Average all TTA predictions for this model
                    tta_averaged_output = torch.stack(tta_predictions).mean(dim=0)
                    batch_ensemble_preds.append(tta_averaged_output.cpu().numpy())
                else:
                    # If TTA is disabled, only the base prediction is used
                    batch_ensemble_preds.append(tta_predictions[0].cpu().numpy())

            # Average the predictions from all models for this batch
            batch_ensemble_preds = np.array(batch_ensemble_preds) # Shape: (Num_Models, Batch_Size, Num_Targets)
            avg_preds = batch_ensemble_preds.mean(axis=0) # Shape: (Batch_Size, Num_Targets)

            all_predictions.append(avg_preds)
            image_ids.extend(img_ids)
    
    final_predictions = np.vstack(all_predictions)
    
    return final_predictions, image_ids

def generate_submission(predictions, image_ids, test_df, target_names):
    """Generate submission file"""
    
    print("\n" + "="*60)
    print("Generating submission...")
    print("="*60)
    
    # Map predictions to image_id
    pred_map = {}
    for i, img_id in enumerate(image_ids):
        for j, target_name in enumerate(target_names):
            sample_id = f"{img_id}__{target_name}"
            # Ensure non-negative predictions, as biomass cannot be negative
            pred_map[sample_id] = max(0.0, predictions[i, j]) 
    
    # Create submission dataframe in the required 'sample_id', 'target' format
    submission_data = []
    for _, row in test_df.iterrows():
        sample_id = row['sample_id']
        submission_data.append({
            'sample_id': sample_id,
            'target': pred_map.get(sample_id, 0.0) 
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(Config.OUTPUT_PATH, index=False)
    
    print(f"✓ Submission saved to: {Config.OUTPUT_PATH}")
    print(f"Shape: {submission_df.shape}")
    print("\nPrediction statistics:")
    print(submission_df['target'].describe())
    
    return submission_df

def run_inference():
    """Main inference function"""
    
    print("="*60)
    print("CSIRO - Image2Biomass Inference")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Ensemble Folds: {Config.FOLDS_TO_INFER}")
    
    try:
        ensemble_info, target_names = load_models_for_ensemble(Config.FOLDS_TO_INFER)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)
    
    # Load the test CSV, which is in long format
    test_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'test.csv'))
    
    # Get unique images from the long-format test data for image loading
    test_df['image_id'] = test_df['sample_id'].apply(extract_image_id)
    test_unique = test_df[['image_id', 'image_path']].drop_duplicates().reset_index(drop=True)
    
    print(f"Test data shape: {test_df.shape}")
    print(f"Unique test images: {len(test_unique)}")
    
    # Create dataset and loader
    test_dataset = BiomassTestDataset(test_unique, Config.DATA_DIR, get_transforms_tta())
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Make Predictions
    predictions, image_ids = predict(ensemble_info, test_loader)
    print(f"Final predictions shape: {predictions.shape}")
    
    # Generate Submission
    generate_submission(
        predictions, 
        image_ids, 
        test_df, 
        target_names
    )
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)

if __name__ == '__main__':
    run_inference()
