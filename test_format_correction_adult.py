"""
Test script for Format Corrector
Tests format correction on Adult dataset
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.format_correction.format_corrector import FormatCorrector


def load_adult_dataset():

    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv(
        'datasets/adult/adult.data',
        names=column_names,
        header=None,
        na_values=' ?',
        skipinitialspace=True
    )
    
    return df


def main():
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("=" * 70)
    print("FORMAT CORRECTION TEST - ADULT DATASET".center(70))
    print("=" * 70)
    
    # Load dataset
    print("\n📋 Loading Adult dataset...")
    df = load_adult_dataset()
    print(f"   Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print("\n🔍 ORIGINAL DATA (First 5 rows)")
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    
    # Initialize corrector
    corrector = FormatCorrector()
    
    # Apply corrections with default config
    config = {
        'normalize_strings': True,
        'standardize_dates': True,
        'correct_types': True,
        'string_case': 'lower'  # Lowercase for consistency
    }
    
    print("\n🔧 Applying format corrections...")
    df_corrected = corrector.correct_formats(df, config)
    
    print("\n✅ CORRECTED DATA (First 5 rows)")
    print(df_corrected.head())
    print(f"\nData types:\n{df_corrected.dtypes}")
    
    # Print summary
    print()
    corrector.print_summary()
    
    # Save corrections log to JSON
    print("\n💾 Saving corrections log...")
    corrector.save_corrections_log('data/adult_format_corrections_log.json')
    
    # Save corrected dataset
    print("\n💾 Saving corrected dataset...")
    df_corrected.to_csv('data/adult_corrected.csv', index=False)
    print(f"   ✓ Corrected dataset saved to data/adult_corrected.csv")
    
    # Summary statistics
    print("\n📊 CORRECTION SUMMARY")
    print("=" * 70)
    print(f"   Total rows processed: {len(df):,}")
    summary = corrector.get_corrections_summary()
    if not summary.empty:
        print(f"   Total columns corrected: {summary['column'].nunique()}")
        print(f"   Total row-level changes: {summary['rows_affected'].sum():,}")
    else:
        print("   No corrections were needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
