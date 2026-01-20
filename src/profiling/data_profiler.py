"""
Data Profiler Module
Generates comprehensive dataset overview and statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class DataProfiler:
    """
    Generates comprehensive dataset profile including:
    - Dataset-level statistics (rows, columns, dtypes, missing %, duplicates)
    - Numeric column statistics (mean, median, std, min, max, correlations)
    - Categorical column statistics (cardinality, mode, distributions)
    """
    
    def __init__(self, additional_na_values: Optional[List[str]] = None):
        """
        Initialize profiler
        
        Args:
            additional_na_values: List of strings to treat as missing values
                                 (e.g., [' ?', '?', 'NA', 'N/A', 'unknown'])
        """
        self.profile = {}
        #default missing value indicators
        self.na_values = [' ?', '?', 'NA', 'N/A', 'na', 'n/a', 
                         'NULL', 'null', 'None', 'none', '', ' ', 
                         'unknown', 'Unknown', 'UNKNOWN', 'missing', 'Missing', '-']
        
        if additional_na_values:
            self.na_values.extend(additional_na_values)
    
    def count_missing(self, series: pd.Series) -> int:
        """
        Count missing values including NaN and other representations
        
        Args:
            series: pandas Series to check
            
        Returns:
            Count of missing values
        """
        #count NaN values
        nan_count = series.isnull().sum()
        
        #count other missing value representations (for object/string columns)
        if series.dtype == 'object':
            for na_val in self.na_values:
                nan_count += (series == na_val).sum()
        
        return int(nan_count)
    
    #main method - generates full profile
    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        self.profile = {
            'dataset_level': self.profile_dataset(df),
            'numeric_columns': self.profile_numeric(df),
            'categorical_columns': self.profile_categorical(df)
        }
        
        return self.profile
    
    #dataset level stats
    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
    
        #basic shape
        n_rows, n_cols = df.shape
        
        #data types
        dtypes = df.dtypes.astype(str).to_dict()
        
        #missing values (enhanced detection)
        missing_counts = {}
        missing_percentages = {}
        non_null_counts = {}
        
        for col in df.columns:
            missing_count = self.count_missing(df[col])
            missing_counts[col] = missing_count
            missing_percentages[col] = round((missing_count / len(df) * 100), 2) if len(df) > 0 else 0
            non_null_counts[col] = len(df) - missing_count
        
        #duplicates (rows only, not columns)
        n_duplicates = df.duplicated().sum()
        duplicate_percentage = round((n_duplicates / len(df) * 100), 2) if len(df) > 0 else 0
        
        #memory
        memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  #MB
        
        return {
            'n_rows': int(n_rows),
            'n_columns': int(n_cols),
            'dtypes': dtypes,
            'non_null_counts': {k: int(v) for k, v in non_null_counts.items()},
            'missing_counts': {k: int(v) for k, v in missing_counts.items()},
            'missing_percentages': {k: round(float(v), 2) for k, v in missing_percentages.items()},
            'n_duplicates': int(n_duplicates),
            'duplicate_percentage': round(float(duplicate_percentage), 2),
            'memory_mb': round(float(memory_usage), 2)
        }
    
    #numeric column stats
    def profile_numeric(self, df: pd.DataFrame) -> Dict[str, Any]:
      
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'columns': {}, 'correlation_matrix': {}}
        
        numeric_profile = {}
        
        for col in numeric_cols:
            #calculate skewness
            skew_val = df[col].skew()
            
            #use custom missing count
            non_null_count = len(df) - self.count_missing(df[col])
            
            #check if column is integer type
            is_integer_type = pd.api.types.is_integer_dtype(df[col])
            
            #for min/max/quartiles/median: preserve original precision
            #for mean/std/skewness: always use 4 decimals
            if not df[col].isna().all():
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                q25_val = float(df[col].quantile(0.25))
                q75_val = float(df[col].quantile(0.75))
                median_val = float(df[col].median())
                
                #if integer type, convert back to int if they're whole numbers
                if is_integer_type:
                    min_val = int(min_val) if min_val == int(min_val) else min_val
                    max_val = int(max_val) if max_val == int(max_val) else max_val
                    q25_val = int(q25_val) if q25_val == int(q25_val) else q25_val
                    q75_val = int(q75_val) if q75_val == int(q75_val) else q75_val
                    median_val = int(median_val) if median_val == int(median_val) else median_val
            else:
                min_val = max_val = q25_val = q75_val = median_val = None
            
            numeric_profile[col] = {
                'count': int(non_null_count),  #non-null count
                'mean': round(float(df[col].mean()), 4) if not df[col].isna().all() else None,
                'median': median_val,
                'std': round(float(df[col].std()), 4) if not df[col].isna().all() else None,
                'min': min_val,
                'max': max_val,
                'q25': q25_val,
                'q75': q75_val,
                'skewness': round(float(skew_val), 4) if pd.notna(skew_val) else None  #type: ignore #distribution shape
            }
        
        #correlation matrix
        correlation_matrix = df[numeric_cols].corr().round(4)
        
        return {
            'columns': numeric_profile,
            'correlation_matrix': correlation_matrix
        }
    
    #categorical column stats
    def profile_categorical(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {'columns': {}}
        
        categorical_profile = {}
        
        for col in categorical_cols:
            #use custom missing count
            non_null_count = len(df) - self.count_missing(df[col])
            
            #calculate cardinality excluding missing values
            unique_count = df[col].nunique()
            for na_val in self.na_values:
                if na_val in df[col].values:
                    unique_count -= 1
            
            categorical_profile[col] = {
                'count': int(non_null_count),  #non-null count
                'cardinality': int(max(0, unique_count)),
                'mode': str(df[col].mode()[0]) if not df[col].mode().empty else None
            }
        
        return {
            'columns': categorical_profile
        }
    
    #summary of dataset
    def print_summary(self):
        
        if not self.profile:
            print("No profile generated yet. Run generate_profile() first.")
            return
        
        dataset = self.profile['dataset_level']
        
        print("=" * 70)
        print("DATASET PROFILE SUMMARY".center(70))
        print("=" * 70)
        
        #dataset level
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   {'Rows:':<20} {dataset['n_rows']:>10,}")
        print(f"   {'Columns:':<20} {dataset['n_columns']:>10}")
        print(f"   {'Duplicate Rows:':<20} {dataset['n_duplicates']:>10} ({dataset['duplicate_percentage']:.2f}%)")       
        
        #missing values summary
        total_missing = sum(dataset['missing_counts'].values())
        missing_cols = len([v for v in dataset['missing_percentages'].values() if v > 0])
        print(f"   {'Missing Values:':<20} {total_missing:>10} ({missing_cols} columns affected)")
        
        print(f"   {'Memory Usage:':<20} {dataset['memory_mb']:>10} MB") 

        #data types
        print(f"\n📋 COLUMN DATA TYPES")
        print(f"   {'Column':<20} {'Type':<15} {'Non-Null Count':<15}")
        print(f"   {'-'*20} {'-'*15} {'-'*15}")
        for col, dtype in dataset['dtypes'].items():
            non_null = dataset['non_null_counts'][col]
            print(f"   {col:<20} {dtype:<15} {non_null:<15,}")
        
        #missing values detail (if any)
        missing = {k: v for k, v in dataset['missing_percentages'].items() if v > 0}
        if missing:
            print(f"\n❓ MISSING VALUES DETAIL")
            print(f"   {'Column':<20} {'Missing':<15} {'Percentage':<15}")
            print(f"   {'-'*20} {'-'*15} {'-'*15}")
            for col, pct in sorted(missing.items(), key=lambda x: x[1], reverse=True):
                print(f"   {col:<20} {dataset['missing_counts'][col]:<15,} {pct:<15.2f}%")
       
        
        
        #numeric columns
        if self.profile['numeric_columns']['columns']:
            print(f"\n🔢 NUMERIC COLUMNS ({len(self.profile['numeric_columns']['columns'])} total)")
            print(f"   {'Column':<20} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Q25':<12} {'Q75':<12} {'Max':<12} {'Skewness':<12}")
            print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
            for col, stats in self.profile['numeric_columns']['columns'].items():
                if stats['mean'] is not None:
                    #format helper: if value is int, show as int; else show with appropriate decimals
                    def format_val(val, calculated=False):
                        if val is None:
                            return "N/A"
                        if isinstance(val, int):
                            return str(val)
                        if calculated:  #mean, std, skewness always show 4 decimals
                            return f"{val:.4f}"
                        return str(val)  #min/max/quartiles show as-is
                    
                    mean_str = format_val(stats['mean'], calculated=True)
                    median_str = format_val(stats['median'])
                    std_str = format_val(stats['std'], calculated=True)
                    min_str = format_val(stats['min'])
                    q25_str = format_val(stats['q25'])
                    q75_str = format_val(stats['q75'])
                    max_str = format_val(stats['max'])
                    skew_str = format_val(stats['skewness'], calculated=True)
                    
                    print(f"   {col:<20} {mean_str:<12} {median_str:<12} {std_str:<12} {min_str:<12} {q25_str:<12} {q75_str:<12} {max_str:<12} {skew_str:<12}")
            
            #correlation matrix
            if 'correlation_matrix' in self.profile['numeric_columns']:
                corr_matrix = self.profile['numeric_columns']['correlation_matrix']
                if hasattr(corr_matrix, 'shape') and corr_matrix.shape[0] > 1:
                    print(f"\n🔗 CORRELATION MATRIX")
                    print(corr_matrix.to_string())
        
        #categorical columns
        if self.profile['categorical_columns']['columns']:
            print(f"\n📝 CATEGORICAL COLUMNS ({len(self.profile['categorical_columns']['columns'])} total)")
            print(f"   {'Column':<20} {'Unique Values':<15} {'Mode':<30}")
            print(f"   {'-'*20} {'-'*15} {'-'*30}")
            for col, stats in self.profile['categorical_columns']['columns'].items():
                mode_str = str(stats['mode'])[:28] if stats['mode'] else 'None'
                print(f"   {col:<20} {stats['cardinality']:<15} {mode_str:<30}")
        
        print("\n" + "=" * 70)
    
    #save profiling report to json file
    def save_report(self, filepath: str):
        
        import json
        
        if not self.profile:
            print("No profile generated yet. Run generate_profile() first.")
            return
        
        #convert non-serializable objects
        profile_copy = self.profile.copy()
        if 'correlation_matrix' in profile_copy.get('numeric_columns', {}):
            corr_matrix = profile_copy['numeric_columns']['correlation_matrix']
            if hasattr(corr_matrix, 'to_dict'):
                profile_copy['numeric_columns']['correlation_matrix'] = corr_matrix.round(4).to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(profile_copy, f, indent=2, default=str)
        
        print(f"✓ Profile saved to {filepath}")
