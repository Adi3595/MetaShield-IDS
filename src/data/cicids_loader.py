"""
CIC-IDS2017 Dataset Loader
===========================
Robust loader for the CIC-IDS2017 intrusion detection dataset.
Handles NaN/Inf values, feature normalization, label encoding,
and base/novel class splitting for few-shot class-incremental learning.

Reference: Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization."
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# CIC-IDS2017 attack type mapping (consolidate sub-categories)
ATTACK_MAPPING = {
    'BENIGN': 'Benign',
    'Bot': 'Bot',
    'DDoS': 'DDoS',
    'DoS GoldenEye': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS slowloris': 'DoS',
    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
    'Heartbleed': 'Heartbleed',
    'Infiltration': 'Infiltration',
    'PortScan': 'PortScan',
    'Web Attack \x96 Brute Force': 'WebAttack',
    'Web Attack \x96 XSS': 'WebAttack',
    'Web Attack \x96 Sql Injection': 'WebAttack',
    'Web Attack – Brute Force': 'WebAttack',
    'Web Attack – XSS': 'WebAttack',
    'Web Attack – Sql Injection': 'WebAttack',
    'Web Attack - Brute Force': 'WebAttack',
    'Web Attack - XSS': 'WebAttack',
    'Web Attack - Sql Injection': 'WebAttack',
}

# Base classes (seen during meta-training) vs Novel classes (for incremental learning)
BASE_CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'DDoS']
NOVEL_CLASSES = ['Bot', 'WebAttack', 'Infiltration', 'Heartbleed']


class CICIDS2017Loader:
    """
    Comprehensive loader for CIC-IDS2017 dataset.
    
    Supports:
    - Loading and merging all 8 CSV files
    - Cleaning NaN/Inf values
    - Feature normalization
    - Label consolidation
    - Base/Novel class splitting for FSCIL
    - Caching processed data for fast reload
    """
    
    def __init__(self, data_dir='data/archive', cache_dir='data/processed',
                 max_samples_per_class=50000, random_state=42):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.max_samples_per_class = max_samples_per_class
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        self.n_features = None
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_and_preprocess(self, use_cache=True):
        """
        Load, clean, and preprocess the full CIC-IDS2017 dataset.
        
        Returns:
            X_train, X_test, y_train, y_test, class_names
        """
        cache_path = os.path.join(self.cache_dir, 'cicids2017_processed.pkl')
        
        if use_cache and os.path.exists(cache_path):
            logger.info("Loading preprocessed data from cache...")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            self.scaler = cached['scaler']
            self.label_encoder = cached['label_encoder']
            self.feature_names = cached['feature_names']
            self.class_names = cached['class_names']
            self.n_features = cached['n_features']
            logger.info(f"Loaded {len(cached['X_train'])} train, {len(cached['X_test'])} test samples")
            return (cached['X_train'], cached['X_test'], 
                    cached['y_train'], cached['y_test'],
                    cached['class_names'])
        
        # Load raw CSV files
        logger.info("Loading raw CIC-IDS2017 CSV files...")
        df = self._load_all_csvs()
        
        # Clean data
        logger.info("Cleaning data...")
        df = self._clean_data(df)
        
        # Map labels
        logger.info("Mapping attack labels...")
        df = self._map_labels(df)
        
        # Balance classes
        logger.info("Balancing classes...")
        df = self._balance_classes(df)
        
        # Extract features and labels
        label_col = ' Label' if ' Label' in df.columns else 'Label'
        y_raw = df[label_col].values
        X_raw = df.drop(columns=[label_col]).values.astype(np.float32)
        
        self.feature_names = [c for c in df.columns if c != label_col]
        self.n_features = X_raw.shape[1]
        
        # Encode labels
        y = self.label_encoder.fit_transform(y_raw)
        self.class_names = list(self.label_encoder.classes_)
        
        # Normalize features
        X = self.scaler.fit_transform(X_raw).astype(np.float32)
        
        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test, "
                     f"{self.n_features} features, {len(self.class_names)} classes")
        logger.info(f"Classes: {self.class_names}")
        
        # Cache
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'n_features': self.n_features
            }, f)
        logger.info(f"Cached processed data to {cache_path}")
        
        return X_train, X_test, y_train, y_test, self.class_names
    
    def get_base_novel_split(self, X_train, y_train, X_test, y_test):
        """
        Split data into base and novel classes for few-shot class-incremental learning.
        
        Base classes: Used during meta-training
        Novel classes: Introduced incrementally during evaluation
        
        Returns:
            base_data, novel_data (each a dict with X_train, X_test, y_train, y_test, classes)
        """
        base_indices_map = {}
        novel_indices_map = {}
        
        for class_name in self.class_names:
            class_idx = self.label_encoder.transform([class_name])[0]
            if class_name in BASE_CLASSES:
                base_indices_map[class_name] = class_idx
            elif class_name in NOVEL_CLASSES:
                novel_indices_map[class_name] = class_idx
        
        # Base data
        base_class_ids = list(base_indices_map.values())
        base_train_mask = np.isin(y_train, base_class_ids)
        base_test_mask = np.isin(y_test, base_class_ids)
        
        # Re-label base classes to 0..N_base-1
        base_label_map = {old: new for new, old in enumerate(sorted(base_class_ids))}
        
        base_y_train = np.array([base_label_map[y] for y in y_train[base_train_mask]])
        base_y_test = np.array([base_label_map[y] for y in y_test[base_test_mask]])
        base_class_names = [self.class_names[idx] for idx in sorted(base_class_ids)]
        
        base_data = {
            'X_train': X_train[base_train_mask],
            'X_test': X_test[base_test_mask],
            'y_train': base_y_train,
            'y_test': base_y_test,
            'classes': base_class_names,
            'label_map': base_label_map
        }
        
        # Novel data
        novel_class_ids = list(novel_indices_map.values())
        novel_train_mask = np.isin(y_train, novel_class_ids)
        novel_test_mask = np.isin(y_test, novel_class_ids)
        
        novel_label_map = {old: new for new, old in enumerate(sorted(novel_class_ids))}
        
        novel_y_train = np.array([novel_label_map[y] for y in y_train[novel_train_mask]]) if novel_train_mask.any() else np.array([])
        novel_y_test = np.array([novel_label_map[y] for y in y_test[novel_test_mask]]) if novel_test_mask.any() else np.array([])
        novel_class_names = [self.class_names[idx] for idx in sorted(novel_class_ids)]
        
        novel_data = {
            'X_train': X_train[novel_train_mask] if novel_train_mask.any() else np.array([]),
            'X_test': X_test[novel_test_mask] if novel_test_mask.any() else np.array([]),
            'y_train': novel_y_train,
            'y_test': novel_y_test,
            'classes': novel_class_names,
            'label_map': novel_label_map
        }
        
        logger.info(f"Base classes ({len(base_class_names)}): {base_class_names}")
        logger.info(f"Novel classes ({len(novel_class_names)}): {novel_class_names}")
        
        return base_data, novel_data
    
    def _load_all_csvs(self):
        """Load and concatenate all CIC-IDS2017 CSV files."""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                f"Please place the CIC-IDS2017 CSV files there."
            )
        
        dfs = []
        for csv_file in sorted(csv_files):
            filepath = os.path.join(self.data_dir, csv_file)
            logger.info(f"  Loading {csv_file}...")
            try:
                df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                dfs.append(df)
                logger.info(f"    → {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"    ⚠ Failed to load {csv_file}: {e}")
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _clean_data(self, df):
        """Clean the dataset: handle NaN, Inf, duplicates, etc."""
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Identify label column
        label_col = 'Label' if 'Label' in df.columns else ' Label'
        if label_col not in df.columns:
            # Try to find it
            label_candidates = [c for c in df.columns if 'label' in c.lower()]
            if label_candidates:
                label_col = label_candidates[0]
            else:
                raise ValueError("Cannot find label column in dataset")
        
        # Rename to consistent name
        df = df.rename(columns={label_col: ' Label'})
        label_col = ' Label'
        
        # Strip whitespace from labels
        df[label_col] = df[label_col].astype(str).str.strip()
        
        # Get feature columns (exclude label)
        feature_cols = [c for c in df.columns if c != label_col]
        
        # Convert features to numeric
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN labels
        df = df.dropna(subset=[label_col])
        
        # Replace Inf with NaN, then fill NaN with column median
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Drop columns that are entirely NaN
        nan_cols = df[feature_cols].columns[df[feature_cols].isna().all()]
        if len(nan_cols) > 0:
            logger.info(f"  Dropping {len(nan_cols)} all-NaN columns")
            df = df.drop(columns=nan_cols)
            feature_cols = [c for c in df.columns if c != label_col]
        
        # Fill remaining NaN with column median
        for col in feature_cols:
            median_val = df[col].median()
            if np.isnan(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
        
        # Drop duplicate rows
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            logger.info(f"  Removed {initial_len - len(df)} duplicate rows")
        
        # Drop constant columns (zero variance)
        constant_cols = [c for c in feature_cols if df[c].std() == 0]
        if constant_cols:
            logger.info(f"  Dropping {len(constant_cols)} constant columns")
            df = df.drop(columns=constant_cols)
        
        logger.info(f"  Cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _map_labels(self, df):
        """Map fine-grained attack labels to consolidated categories."""
        label_col = ' Label'
        
        # Show original distribution
        original_dist = df[label_col].value_counts()
        logger.info(f"  Original label distribution:\n{original_dist.to_string()}")
        
        # Map labels
        df[label_col] = df[label_col].map(
            lambda x: ATTACK_MAPPING.get(x.strip(), x.strip())
        )
        
        # Remove any classes with very few samples (< 10)
        class_counts = df[label_col].value_counts()
        small_classes = class_counts[class_counts < 10].index.tolist()
        if small_classes:
            logger.info(f"  Removing classes with <10 samples: {small_classes}")
            df = df[~df[label_col].isin(small_classes)]
        
        mapped_dist = df[label_col].value_counts()
        logger.info(f"  Mapped label distribution:\n{mapped_dist.to_string()}")
        
        return df
    
    def _balance_classes(self, df):
        """Balance classes by capping large classes and keeping small ones."""
        label_col = ' Label'
        
        balanced_dfs = []
        for class_name in df[label_col].unique():
            class_df = df[df[label_col] == class_name]
            if len(class_df) > self.max_samples_per_class:
                class_df = class_df.sample(
                    n=self.max_samples_per_class, 
                    random_state=self.random_state
                )
            balanced_dfs.append(class_df)
        
        df = pd.concat(balanced_dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"  Balanced dataset: {len(df)} samples")
        return df
