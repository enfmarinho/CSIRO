import torch

class Config:
    # Paths
    DATA_DIR = 'dataset'
    OUTPUT_DIR = 'checkpoints'
    
    # Model
    BACKBONE_MODEL = 'convnext_small'
    IMG_SIZE = 512
    
    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 150
    LEARNING_RATE = 5e-5
    PATIENCE = 10
    NUM_WORKERS = 4
    
    # Cross-validation
    NUM_FOLDS = 5
    TRAIN_FOLD = 0  # Which fold to train, or use -1 to train all folds. Used for testing
    
    # Multi-task learning
    USE_MULTITASK = True
    AUXILIARY_WEIGHT = 0.3
    
    # Data Augmentation Strategy
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    
    # Targets
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2]

    # Other
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
