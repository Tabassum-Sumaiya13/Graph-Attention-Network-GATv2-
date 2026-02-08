"""
Data loading utilities for the GNN Survival Pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_expression_data(data_dir: Path) -> pd.DataFrame:
    """Load expression data with fallback options"""
    # Try parquet first (fastest)
    parquet_path = data_dir / "labeled_arcsinh_norm_data.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    
    # Try CSV
    csv_path = data_dir / "labeled_arcsinh_norm_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Try pickle (slowest, may have compatibility issues)
    pkl_path = data_dir / "labeled_arcsinh_norm_data.pkl"
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    
    raise FileNotFoundError(f"No expression data found in {data_dir}")


def load_all_data(config) -> Tuple[pd.DataFrame, pd.DataFrame, list, list, int, pd.DataFrame]:
    """
    Load all required data files.
    
    Returns:
        sample_df: Sample metadata with survival information
        expr: Expression data (arcsinh normalized)
        marker_names: List of marker names
        cluster_names: List of cluster/cell type names
        num_clusters: Number of unique clusters
        cell_locs: Cell location data with X, Y coordinates
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_dir = config.DATA_DIR
    
    # 1. Sample metadata
    sample_df = pd.read_csv(data_dir / "sample_metadata.csv")
    
    # 2. QC-passing samples
    qc_file = data_dir / "qc_acq_ids_labeled.csv"
    if qc_file.exists():
        qc_samples = pd.read_csv(qc_file, header=None).iloc[:, 0].values
        sample_df = sample_df[sample_df.acquisition_id.isin(qc_samples)]
    
    # 3. Expression data
    expr = load_expression_data(data_dir)
    print(f"  Raw expression data: {expr.shape[0]:,} cells, {expr.shape[1]} columns")
    
    # 4. Marker names
    marker_names = list(pd.read_csv(data_dir / "marker_names.csv").iloc[:, 0].values)
    
    # 5. Cluster information
    cluster_names = sorted(expr.cluster_label.unique())
    num_clusters = len(cluster_names)
    
    # 6. Cell locations
    cell_locs = pd.read_csv(data_dir / "cell_locations_and_labels.csv")
    
    # 7. Subsample if needed
    if config.MAX_SAMPLES and len(sample_df) > config.MAX_SAMPLES:
        # Keep at least 2 samples per patient for proper CV
        sample_df = sample_df.groupby('patient_id', group_keys=False).apply(
            lambda x: x.head(2)
        ).head(config.MAX_SAMPLES)
    
    # Summary
    print(f"  ✓ Samples: {len(sample_df)} from {sample_df.patient_id.nunique()} patients")
    print(f"  ✓ Total cells: {expr.shape[0]:,}")
    print(f"  ✓ Markers: {len(marker_names)}")
    print(f"  ✓ Cell types: {num_clusters}")
    print(f"  ✓ Event rate: {sample_df.survival_status.mean():.1%}")
    
    return sample_df, expr, marker_names, cluster_names, num_clusters, cell_locs


def get_sample_data(sample_id: str, expr: pd.DataFrame, cell_locs: pd.DataFrame,
                    config) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Get expression and location data for a single sample.
    
    Returns:
        sample_expr: Expression data for the sample (or None if insufficient cells)
        sample_locs: Location data for the sample (or None if insufficient cells)
    """
    # Get cells for this sample
    sample_expr = expr[expr.sample_id == sample_id].copy()
    sample_locs = cell_locs[cell_locs.ACQUISITION_ID == sample_id].copy()
    
    # Check minimum cells
    if len(sample_expr) < config.MIN_CELLS_PER_SAMPLE:
        return None, None
    
    if len(sample_locs) < config.MIN_CELLS_PER_SAMPLE:
        return None, None
    
    # Subsample if too many cells (for memory management)
    if len(sample_expr) > config.MAX_CELLS_PER_SAMPLE:
        np.random.seed(config.RANDOM_STATE)
        idx_sample = np.random.choice(len(sample_expr), config.MAX_CELLS_PER_SAMPLE, replace=False)
        sample_expr = sample_expr.iloc[idx_sample].reset_index(drop=True)
        sample_locs = sample_locs.iloc[idx_sample].reset_index(drop=True)
    
    return sample_expr, sample_locs
