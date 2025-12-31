"""
Data Cleaning Pipeline
Orchestrates the full data cleaning process
"""

import pandas as pd
import os
from typing import Dict, Any, Optional

from src.profiling.data_profiler import DataProfiler
from src.format_correction.format_corrector import FormatCorrector
from src.duplicates.duplicate_detector import DuplicateDetector


class DataCleaningPipeline:
    """
    Main pipeline that runs all cleaning steps in sequence:
    1. Data Profiling (analyze dataset)
    2. Format Correction (normalize strings, dates, types)
    3. Duplicate Detection (duplicate rows)
    """
    
    def __init__(self, output_dir: str = 'data/output'):
        self.output_dir = output_dir
        self.profiler = DataProfiler()
        self.format_corrector = FormatCorrector()
        self.duplicate_detector = DuplicateDetector()
        
        self.original_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.profile_before: Optional[Dict[str, Any]] = None
        self.profile_after: Optional[Dict[str, Any]] = None
        self.pipeline_log: list = []
        
        #create output directory if doesnt exist
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, df: pd.DataFrame, dataset_name: str = 'dataset', 
            config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Run the full cleaning pipeline
        
        Args:
            df: Input dataframe
            dataset_name: Name for output files
            config: Optional configuration dict
        
        Returns:
            Cleaned dataframe
        """
        self.original_df = df.copy()
        self.cleaned_df = df.copy()
        self.pipeline_log = []
        
        #default config
        if config is None:
            config = {
                'profiling': True,
                'format_correction': {
                    'normalize_strings': True,
                    'standardize_dates': True,
                    'correct_types': True,
                    'string_case': 'lower'
                },
                'duplicate_detection': True,  #coming later
            }
        
        print("=" * 70)
        print("DATA CLEANING PIPELINE".center(70))
        print("=" * 70)
        print(f"\n📁 Dataset: {dataset_name}")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        #step 1: initial profiling
        if config.get('profiling', True):
            print("\n" + "-" * 70)
            print("STEP 1: DATA PROFILING (Before Cleaning)".center(70))
            print("-" * 70)
            
            self.profile_before = self.profiler.generate_profile(self.cleaned_df)
            self.profiler.print_summary()
            self.profiler.save_report(f"{self.output_dir}/{dataset_name}_profile_before.json")
            
            self.log_step('profiling_before', {
                'rows': self.profile_before['dataset_level']['n_rows'],
                'columns': self.profile_before['dataset_level']['n_columns'],
                'missing_total': sum(self.profile_before['dataset_level']['missing_counts'].values()),
                'duplicates': self.profile_before['dataset_level']['n_duplicates']
            })
        
        #step 2: format correction
        if config.get('format_correction'):
            print("\n" + "-" * 70)
            print("STEP 2: FORMAT CORRECTION".center(70))
            print("-" * 70)
            
            format_config = config['format_correction']
            self.cleaned_df = self.format_corrector.correct_formats(self.cleaned_df, format_config)
            self.format_corrector.print_summary()
            self.format_corrector.save_corrections_log(f"{self.output_dir}/{dataset_name}_format_corrections.json")
            
            summary = self.format_corrector.get_corrections_summary()
            self.log_step('format_correction', {
                'columns_corrected': summary['column'].nunique() if not summary.empty else 0,
                'total_changes': int(summary['rows_affected'].sum()) if not summary.empty else 0
            })
        
        #step 3: duplicate detection & removal
        if config.get('duplicate_detection', True):
            print("\n" + "-" * 70)
            print("STEP 3: DUPLICATE DETECTION".center(70))
            print("-" * 70)
            
            dup_config = config.get('duplicate_config', {})
            subset = dup_config.get('subset', None)
            keep = dup_config.get('keep', 'first')
            
            #first detect and display duplicates
            duplicates = self.duplicate_detector.detect_duplicates(self.cleaned_df, subset=subset)
            
            if duplicates is not None and len(duplicates) > 0:
                print(f"\n🔍 DUPLICATES FOUND: {len(duplicates)} duplicate rows")
                
                #get duplicate groups (original + duplicates together)
                duplicate_groups = self.duplicate_detector.get_duplicate_groups(self.cleaned_df, subset=subset)
                
                if duplicate_groups is not None and len(duplicate_groups) > 0:
                    print(f"\n📋 DUPLICATE GROUPS (showing first 3 groups):")
                    print("   Each group shows the original row and its duplicate(s)")
                    print("-" * 70)
                    
                    #group by all columns to show duplicates together
                    cols = subset if subset else self.cleaned_df.columns.tolist()
                    grouped = duplicate_groups.groupby(cols, dropna=False)
                    
                    group_count = 0
                    for name, group in grouped:
                        if group_count >= 3:  #show first 3 groups
                            break
                        print(f"\n   GROUP {group_count + 1} ({len(group)} identical rows):")
                        print(group.to_string())
                        group_count += 1
                    
                    print("-" * 70)
                
                #now proceed to removal
                print(f"\n🗑️  REMOVING DUPLICATES...")
                self.cleaned_df = self.duplicate_detector.remove_duplicates(self.cleaned_df, subset=subset, keep=keep)
                print(f"   ✓ Removed {self.duplicate_detector.duplicates_removed} duplicate rows (keeping '{keep}' occurrence)")
                print(f"   ✓ Remaining rows: {len(self.cleaned_df):,}")
            else:
                print(f"\n✓ No duplicates found in dataset")
            
            self.duplicate_detector.save_duplicates_log(f"{self.output_dir}/{dataset_name}_duplicates.json")
            
            self.log_step('duplicate_detection', {
                'duplicates_found': self.duplicate_detector.duplicates_found,
                'duplicates_removed': self.duplicate_detector.duplicates_removed
            })
        
        #final profiling
        if config.get('profiling', True):
            print("\n" + "-" * 70)
            print("FINAL: DATA PROFILING (After Cleaning)".center(70))
            print("-" * 70)
            
            #re-initialize profiler for clean state
            final_profiler = DataProfiler()
            self.profile_after = final_profiler.generate_profile(self.cleaned_df)
            final_profiler.print_summary()
            final_profiler.save_report(f"{self.output_dir}/{dataset_name}_profile_after.json")
            
            self.log_step('profiling_after', {
                'rows': self.profile_after['dataset_level']['n_rows'],
                'columns': self.profile_after['dataset_level']['n_columns'],
                'missing_total': sum(self.profile_after['dataset_level']['missing_counts'].values()),
                'duplicates': self.profile_after['dataset_level']['n_duplicates']
            })
        
        #save cleaned dataset
        output_path = f"{self.output_dir}/{dataset_name}_cleaned.csv"
        self.cleaned_df.to_csv(output_path, index=False)
        
        #print pipeline summary
        self.print_pipeline_summary(dataset_name, output_path)
        
        return self.cleaned_df
    
    def log_step(self, step_name: str, details: Dict[str, Any]):
        """Log a pipeline step"""
        self.pipeline_log.append({
            'step': step_name,
            'details': details
        })
    
    def print_pipeline_summary(self, dataset_name: str, output_path: str):
        """Print final pipeline summary"""
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE".center(70))
        print("=" * 70)
        
        if self.profile_before and self.profile_after:
            before = self.profile_before['dataset_level']
            after = self.profile_after['dataset_level']
            
            print(f"\n{'Metric':<25} {'Before':<15} {'After':<15} {'Change':<15}")
            print("-" * 70)
            
            #rows
            row_change = after['n_rows'] - before['n_rows']
            print(f"{'Rows':<25} {before['n_rows']:<15,} {after['n_rows']:<15,} {row_change:+,}")
            
            #missing values
            missing_before = sum(before['missing_counts'].values())
            missing_after = sum(after['missing_counts'].values())
            missing_change = missing_after - missing_before
            print(f"{'Missing Values':<25} {missing_before:<15,} {missing_after:<15,} {missing_change:+,}")
            
            #duplicates
            dup_change = after['n_duplicates'] - before['n_duplicates']
            print(f"{'Duplicate Rows':<25} {before['n_duplicates']:<15,} {after['n_duplicates']:<15,} {dup_change:+,}")
        
        print("\n📁 Output Files:")
        print(f"   • Cleaned dataset: {output_path}")
        print(f"   • Profile (before): {self.output_dir}/{dataset_name}_profile_before.json")
        print(f"   • Profile (after): {self.output_dir}/{dataset_name}_profile_after.json")
        print(f"   • Format corrections: {self.output_dir}/{dataset_name}_format_corrections.json")
        print(f"   • Duplicates log: {self.output_dir}/{dataset_name}_duplicates.json")
        
        print("\n" + "=" * 70)
