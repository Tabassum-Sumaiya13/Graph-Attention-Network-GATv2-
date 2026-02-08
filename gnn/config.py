"""
Configuration settings for the GNN Survival Pipeline
"""
from pathlib import Path
import torch


class Config:
    """Configuration class for the GNN pipeline"""
    
    # ==== PATHS ====
    DATA_DIR = Path("New folder/dataset_info")
    OUTPUT_DIR = Path("output/gnn_pipeline")
    PRECOMPUTED_DIR = Path("output/full_run")
    
    # ==== DATA SETTINGS ====
    MAX_SAMPLES = None             # Use ALL samples
    MAX_CELLS_PER_SAMPLE = 800     # Limit cells per graph (memory management)
    MIN_CELLS_PER_SAMPLE = 20      # Minimum cells needed for a sample
    
    # ==== GRAPH CONSTRUCTION ====
    GRAPH_TYPE = 'knn'             # 'knn', 'radius', 'delaunay'
    K_NEIGHBORS = 15               # k for k-NN graph
    RADIUS = 50                    # Radius for radius graph (pixels)
    
    # ==== NODE FEATURES ====
    USE_BIOMARKERS = True          # Include biomarker expression
    USE_CELLTYPE_ONEHOT = True     # Include cell type one-hot encoding
    USE_DENSITY_FEATURES = True    # Include local density features
    USE_SPATIAL_CONTEXT = True     # Include spatial context features
    USE_MORPHOLOGY = False         # Include morphological features (if available)
    DENSITY_K = 15                 # k for density calculations
    CONTEXT_K = 15                 # k for spatial context
    
    # ==== EDGE FEATURES ====
    USE_EDGE_DISTANCE = True       # Edge distance feature
    USE_EDGE_SIMILARITY = True     # Biomarker similarity feature
    USE_EDGE_SAMETYPE = True       # Same cell type indicator
    
    # ==== MODEL ARCHITECTURE ====
    MODEL_TYPE = 'GATv2'           # 'GAT', 'GATv2', 'GCN', 'GraphSAGE'
    HIDDEN_DIM = 128               # Hidden layer dimension
    NUM_HEADS = 8                  # Number of attention heads
    NUM_LAYERS = 4                 # Number of GNN layers
    DROPOUT = 0.2                  # Dropout rate (reduced for better fitting)
    POOLING = 'combined'           # 'mean', 'max', 'add', 'combined', 'attention'
    
    # ==== TRAINING ====
    EPOCHS = 150                   # More epochs for better fitting
    BATCH_SIZE = 16                # Batch size
    LEARNING_RATE = 0.001          # Higher initial learning rate
    WEIGHT_DECAY = 5e-5            # Less regularization for better fitting
    PATIENCE = 20                  # Early stopping patience
    MIN_LR = 1e-6                  # Minimum learning rate
    LR_FACTOR = 0.5                # LR reduction factor
    GRADIENT_CLIP = 1.0            # Gradient clipping
    
    # ==== CROSS-VALIDATION ====
    N_SPLITS = 5                   # Number of CV folds
    STRATIFY_BY = 'patient'        # Stratification: 'patient', 'event', None
    
    # ==== DEVICE ====
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4                # DataLoader workers
    
    # ==== VISUALIZATION ====
    DPI = 150                      # Figure DPI
    FIGSIZE = (16, 12)             # Default figure size
    
    # ==== REPRODUCIBILITY ====
    RANDOM_STATE = 42
    DETERMINISTIC = True
    
    def __init__(self, **kwargs):
        """Override defaults with kwargs"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def __repr__(self):
        return '\n'.join(f'{k}: {v}' for k, v in vars(self).items() 
                        if not k.startswith('_'))
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in vars(self.__class__).items() 
                if not k.startswith('_') and not callable(v)}


# Quick configs for different scenarios
class QuickTestConfig(Config):
    """Fast testing configuration"""
    MAX_SAMPLES = 50
    MAX_CELLS_PER_SAMPLE = 200
    EPOCHS = 20
    N_SPLITS = 3
    HIDDEN_DIM = 32
    NUM_HEADS = 2
    NUM_LAYERS = 2


class FullRunConfig(Config):
    """Full training configuration"""
    MAX_SAMPLES = None  # Use all
    MAX_CELLS_PER_SAMPLE = 1000
    EPOCHS = 200
    N_SPLITS = 5
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 5
