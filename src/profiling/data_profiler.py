"""
Data Profiler Module
Generates comprehensive dataset overview and statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class DataProfiler:
    """
    Generates comprehensive dataset profile including:
    - Dataset-level statistics (rows, columns, dtypes, missing %, duplicates)
    - Numeric column statistics (mean, median, std, min, max, correlations)
    - Categorical column statistics (cardinality, mode, distributions)
    """
    
    def __init__(self):
        self.profile = {}
    
    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method - generates complete profile
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with all profiling information
        """
        self.profile = {
            'dataset_level': self._profile_dataset(df),
            'numeric_columns': self._profile_numeric(df),
            'categorical_columns': self._profile_categorical(df)
        }
        
        return self.profile
    
    def _profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Dataset-level statistics
        """
        # Basic shape
        n_rows, n_cols = df.shape
        
        # Data types
        dtypes = df.dtypes.astype(str).to_dict()
        
        # Missing values
        # Missing values
        missing_counts_series = df.isnull().sum()
        missing_percentages_series = (missing_counts_series / len(df) * 100)
        missing_counts = dict(missing_counts_series)
        missing_percentages = dict(missing_percentages_series)
        
        # Non-null counts (from df.info())
        non_null_counts = df.count().to_dict()
        
        # Duplicates (rows only, not columns)
        n_duplicates = df.duplicated().sum()
        duplicate_percentage = (n_duplicates / len(df) * 100) if len(df) > 0 else 0
        
        # Memory
        memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
        
        return {
            'n_rows': int(n_rows),
            'n_columns': int(n_cols),
            'dtypes': dtypes,
            'non_null_counts': {k: int(v) for k, v in non_null_counts.items()},
            'missing_counts': {k: int(v) for k, v in missing_counts.items()},
            'missing_percentages': {k: float(v) for k, v in missing_percentages.items()},
            'n_duplicates': int(n_duplicates),
            'duplicate_percentage': float(duplicate_percentage),
            'memory_mb': round(float(memory_usage), 2)
        }
    
    def _profile_numeric(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Numeric column statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'columns': {}, 'correlation_matrix': {}}
        
        numeric_profile = {}
        
        for col in numeric_cols:
            # Calculate skewness
            skew_val = df[col].skew()
            
            numeric_profile[col] = {
                'count': int(df[col].count()),  # Non-null count
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'median': float(df[col].median()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'q25': float(df[col].quantile(0.25)) if not df[col].isna().all() else None,
                'q75': float(df[col].quantile(0.75)) if not df[col].isna().all() else None,
                'skewness': float(skew_val) if pd.notna(skew_val) else None  # Distribution shape
            }
        
        # Correlation matrix
        correlation_matrix = df[numeric_cols].corr()
        
        return {
            'columns': numeric_profile,
            'correlation_matrix': correlation_matrix
        }
    
    def _profile_categorical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Categorical column statistics
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {'columns': {}}
        
        categorical_profile = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            categorical_profile[col] = {
                'count': int(df[col].count()),  # Non-null count
                'cardinality': int(df[col].nunique()),
                'mode': str(df[col].mode()[0]) if not df[col].mode().empty else None
            }
        
        return {
            'columns': categorical_profile
        }
    
    def print_summary(self):
        """
        Print human-readable summary to console
        """
        if not self.profile:
            print("No profile generated yet. Run generate_profile() first.")
            return
        
        dataset = self.profile['dataset_level']
        
        print("=" * 70)
        print("DATASET PROFILE SUMMARY".center(70))
        print("=" * 70)
        
        # Dataset level
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   {'Rows:':<20} {dataset['n_rows']:>10,}")
        print(f"   {'Columns:':<20} {dataset['n_columns']:>10}")
        print(f"   {'Memory Usage:':<20} {dataset['memory_mb']:>10} MB")
        print(f"   {'Duplicate Rows:':<20} {dataset['n_duplicates']:>10} ({dataset['duplicate_percentage']:.2f}%)")
        
        # Missing values summary
        total_missing = sum(dataset['missing_counts'].values())
        missing_cols = len([v for v in dataset['missing_percentages'].values() if v > 0])
        print(f"   {'Missing Values:':<20} {total_missing:>10} ({missing_cols} columns affected)")
        
        # Data types
        print(f"\n📋 COLUMN DATA TYPES")
        print(f"   {'Column':<20} {'Type':<15} {'Non-Null Count':<15}")
        print(f"   {'-'*20} {'-'*15} {'-'*15}")
        for col, dtype in dataset['dtypes'].items():
            non_null = dataset['non_null_counts'][col]
            print(f"   {col:<20} {dtype:<15} {non_null:<15,}")
        
        # Missing values detail (if any)
        missing = {k: v for k, v in dataset['missing_percentages'].items() if v > 0}
        if missing:
            print(f"\n❓ MISSING VALUES DETAIL")
            print(f"   {'Column':<20} {'Missing':<15} {'Percentage':<15}")
            print(f"   {'-'*20} {'-'*15} {'-'*15}")
            for col, pct in sorted(missing.items(), key=lambda x: x[1], reverse=True):
                print(f"   {col:<20} {dataset['missing_counts'][col]:<15,} {pct:<15.2f}%")
       
        
        
        
        # Numeric columns
        if self.profile['numeric_columns']['columns']:
            print(f"\n🔢 NUMERIC COLUMNS ({len(self.profile['numeric_columns']['columns'])} total)")
            print(f"   {'Column':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
            for col, stats in self.profile['numeric_columns']['columns'].items():
                if stats['mean'] is not None:
                    print(f"   {col:<20} {stats['mean']:<12.2f} {stats['std']:<12.2f} {stats['min']:<12.2f} {stats['max']:<12.2f}")
            
            # Correlation matrix
            if 'correlation_matrix' in self.profile['numeric_columns']:
                corr_matrix = self.profile['numeric_columns']['correlation_matrix']
                if hasattr(corr_matrix, 'shape') and corr_matrix.shape[0] > 1:
                    print(f"\n🔗 CORRELATION MATRIX")
                    print(corr_matrix.to_string())
        
        # Categorical columns
        if self.profile['categorical_columns']['columns']:
            print(f"\n📝 CATEGORICAL COLUMNS ({len(self.profile['categorical_columns']['columns'])} total)")
            print(f"   {'Column':<20} {'Unique Values':<15} {'Mode':<30}")
            print(f"   {'-'*20} {'-'*15} {'-'*30}")
            for col, stats in self.profile['categorical_columns']['columns'].items():
                mode_str = str(stats['mode'])[:28] if stats['mode'] else 'None'
                print(f"   {col:<20} {stats['cardinality']:<15} {mode_str:<30}")
        
        print("\n" + "=" * 70)
    
    def save_report(self, filepath: str):
        """
        Save profile to JSON file
        
        Args:
            filepath: Path to save JSON report
        """
        import json
        
        if not self.profile:
            print("No profile generated yet. Run generate_profile() first.")
            return
        
        # Convert non-serializable objects
        profile_copy = self.profile.copy()
        if 'correlation_matrix' in profile_copy.get('numeric_columns', {}):
            corr_matrix = profile_copy['numeric_columns']['correlation_matrix']
            if hasattr(corr_matrix, 'to_dict'):
                profile_copy['numeric_columns']['correlation_matrix'] = corr_matrix.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(profile_copy, f, indent=2, default=str)
        
        print(f"✓ Profile saved to {filepath}")
