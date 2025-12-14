import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import Config

def extract_image_id(sample_id):
    """Extract image ID from sample_id"""
    return sample_id.split('__')[0]


def pivot_train_data(train_df):
    """Convert train data from long format to wide format"""
    train_df['image_id'] = train_df['sample_id'].apply(extract_image_id)
    
    # Pivot target values
    targets_wide = train_df.pivot_table(
        index='image_id',
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()
    
    # Get metadata
    metadata_cols = ['image_id', 'image_path', 'Sampling_Date', 'State', 'Species', 
                     'Pre_GSHH_NDVI', 'Height_Ave_cm']
    metadata = train_df[metadata_cols].drop_duplicates('image_id').reset_index(drop=True)
    
    # Merge
    df_wide = metadata.merge(targets_wide, on='image_id', how='left')
    
    return df_wide


def preprocess_data(train_df):
    """Preprocess and encode metadata"""
    df = train_df.copy()
    
    # Handle missing values
    if 'Pre_GSHH_NDVI' in df.columns:
        df['Pre_GSHH_NDVI'].fillna(df['Pre_GSHH_NDVI'].median(), inplace=True)
    if 'Height_Ave_cm' in df.columns:
        df['Height_Ave_cm'].fillna(df['Height_Ave_cm'].median(), inplace=True)
    
    # Scale continuous variables
    scalers = {}
    if 'Pre_GSHH_NDVI' in df.columns:
        scalers['ndvi'] = StandardScaler()
        df['ndvi_scaled'] = scalers['ndvi'].fit_transform(df[['Pre_GSHH_NDVI']])
    
    if 'Height_Ave_cm' in df.columns:
        scalers['height'] = StandardScaler()
        df['height_scaled'] = scalers['height'].fit_transform(df[['Height_Ave_cm']])
    
    # Encode categorical variables
    label_encoders = {}
    for col in ['State', 'Species']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df, scalers, label_encoders

def fit_and_transform_fold_data(train_data, val_data, categorical_cols=['State', 'Species']):
    """
    Fits preprocessing objects (scalers, encoders) only on train_data
    and applies the transformation to both train and val data.
    """
    
    scalers = {}
    label_encoders = {}

    # Imputation
    print("Applying imputation and scaling...")
    
    # Calculate medians for imputation only from the training data
    ndvi_median = train_data['Pre_GSHH_NDVI'].median()
    height_median = train_data['Height_Ave_cm'].median()

    # Apply the training set's medians to both splits
    train_data['Pre_GSHH_NDVI'].fillna(ndvi_median, inplace=True)
    val_data['Pre_GSHH_NDVI'].fillna(ndvi_median, inplace=True)
    train_data['Height_Ave_cm'].fillna(height_median, inplace=True)
    val_data['Height_Ave_cm'].fillna(height_median, inplace=True)

    # Scaling

    # Fit StandardScaler on the training data
    ndvi_scaler = StandardScaler()
    train_data['ndvi_scaled'] = ndvi_scaler.fit_transform(train_data[['Pre_GSHH_NDVI']])
    
    # Apply the learned transformation (without re-fitting) to the validation data
    val_data['ndvi_scaled'] = ndvi_scaler.transform(val_data[['Pre_GSHH_NDVI']])
    scalers['ndvi'] = ndvi_scaler # Save for test set prediction

    height_scaler = StandardScaler()
    train_data['height_scaled'] = height_scaler.fit_transform(train_data[['Height_Ave_cm']])
    val_data['height_scaled'] = height_scaler.transform(val_data[['Height_Ave_cm']])
    scalers['height'] = height_scaler

    # Encoding
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Fit only on the training data's unique values
        le.fit(train_data[col].astype(str).unique())
        
        # Transform training data
        train_data[f'{col}_encoded'] = le.transform(train_data[col].astype(str))
        
        # Transform validation data safely: handle categories that only exist in the validation set
        def safe_transform(s):
            # Create a dictionary mapping known classes to their index
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            # Set unseen categories to a new index
            unseen_idx = len(le.classes_) 
            return [mapping.get(x, unseen_idx) for x in s]
        
        val_data[f'{col}_encoded'] = safe_transform(val_data[col].astype(str))
        
        label_encoders[col] = le
        
    return train_data, val_data, scalers, label_encoders


class BiomassDataset(Dataset):
    """Dataset for training"""
    
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.target_names = Config.TARGET_NAMES
        
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
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        # Biomass targets
        biomass = torch.FloatTensor([
            row.get(target, 0.0) for target in self.target_names
        ])
        
        # Auxiliary targets
        auxiliary = {
            'ndvi': torch.FloatTensor([row.get('ndvi_scaled', 0.0)]),
            'height': torch.FloatTensor([row.get('height_scaled', 0.0)]),
            'state': torch.LongTensor([int(row.get('State_encoded', 0))]),
            'species': torch.LongTensor([int(row.get('Species_encoded', 0))])
        }
        
        return image, biomass, auxiliary


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            # Resize/Crop
            transforms.RandomResizedCrop(
                Config.IMG_SIZE, 
                scale=(0.9, 1.0),    # Only slight zoom variation
                ratio=(0.95, 1.05)   # Keep nearly square
            ),
            
            # Geometric augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            
            # Color
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.05
            ),
            
            # Conversion & Normalization
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
