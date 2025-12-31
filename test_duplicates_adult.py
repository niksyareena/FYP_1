"""
Test script for Duplicate Detector
Tests duplicate detection and removal on Adult dataset
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.duplicates.duplicate_detector import DuplicateDetector


def load_adult_dataset():
    """
    Load the UCI Adult dataset
    """
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
    
    print("=" * 70)
    print("DUPLICATE DETECTION TEST - ADULT DATASET".center(70))
    print("=" * 70)
    
    # Load dataset
    print("\n📋 Loading Adult dataset...")
    df = load_adult_dataset()
    print(f"   Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Initialize detector
    detector = DuplicateDetector()
    
    # Step 1: Detect duplicates
    print("\n🔍 Detecting duplicates...")
    duplicates = detector.detect_duplicates(df)
    num_duplicates = len(duplicates) if duplicates is not None else 0
    print(f"   Found {num_duplicates:,} duplicate rows")
    
    # Step 2: Show duplicate groups (for inspection)
    if num_duplicates > 0:
        print("\n📋 Sample duplicate groups:")
        groups = detector.get_duplicate_groups(df)
        print(groups.head(10))
    
    # Step 3: Remove duplicates
    print("\n🔧 Removing duplicates (keeping first occurrence)...")
    df_cleaned = detector.remove_duplicates(df, keep='first')
    print(f"   Original: {len(df):,} rows")
    print(f"   After removal: {len(df_cleaned):,} rows")
    print(f"   Removed: {len(df) - len(df_cleaned):,} rows")
    
    # Print summary
    print()
    detector.print_summary()
    
    # Save log
    print("\n💾 Saving duplicates log...")
    detector.save_duplicates_log('data/adult_duplicates_log.json')
    
    # Save cleaned dataset
    print("\n💾 Saving cleaned dataset...")
    df_cleaned.to_csv('data/adult_deduplicated.csv', index=False)
    print(f"   ✓ Cleaned dataset saved to data/adult_deduplicated.csv")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
