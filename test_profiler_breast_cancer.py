"""
Test script for Breast Cancer dataset from UCI ML Repository
Dataset: 286 instances, 10 attributes (9 features + 1 class)
Missing values: '?' in node-caps (8) and breast-quad (1)
"""

import pandas as pd
from src.profiling.data_profiler import DataProfiler

#set pandas display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def load_breast_cancer_dataset():
    """
    Load UCI Breast Cancer dataset
    
    Format: CSV with no header
    Columns: class, age, menopause, tumor-size, inv-nodes, node-caps, 
             deg-malig, breast, breast-quad, irradiat
    Missing values: '?'
    """
    
    #define column names based on breast-cancer.names
    column_names = [
        'class',           #no-recurrence-events, recurrence-events
        'age',             #10-19, 20-29, ..., 90-99
        'menopause',       #lt40, ge40, premeno
        'tumor-size',      #0-4, 5-9, ..., 55-59
        'inv-nodes',       #0-2, 3-5, ..., 36-39
        'node-caps',       #yes, no (8 missing)
        'deg-malig',       #1, 2, 3
        'breast',          #left, right
        'breast-quad',     #left-up, left-low, right-up, right-low, central (1 missing)
        'irradiat'         #yes, no
    ]
    
    #load data
    df = pd.read_csv(
        'datasets/breast+cancer/breast-cancer.data',
        names=column_names,
        header=None,
        na_values='?',  #uci format uses '?' for missing values
        skipinitialspace=True
    )
    
    return df


def main():
    print("=" * 70)
    print("BREAST CANCER DATASET TEST".center(70))
    print("=" * 70)
    
    #load dataset
    df = load_breast_cancer_dataset()
    
    print(f"\n📁 DATASET LOADED")
    print(f"   {'Shape:':<20} {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    #display first 5 rows
    print(f"\n" + "=" * 70)
    print("FIRST 5 ROWS".center(70))
    print("=" * 70)
    print(df.head())
    
    #generate profile
    print(f"\n{'='*70}")
    print("GENERATING PROFILE...".center(70))
    print(f"{'='*70}\n")
    
    profiler = DataProfiler()
    profile = profiler.generate_profile(df)
    
    #print summary
    profiler.print_summary()
    
    #save to JSON
    profiler.save_report('data/breast_cancer_profile_report.json')
    
    print(f"\n{'='*70}")
    print("✓ PROFILING COMPLETE".center(70))
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
