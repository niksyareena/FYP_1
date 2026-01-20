"""
Test script for Breast Cancer dataset from UCI ML Repository
Dataset: 286 instances, 10 attributes (9 features + 1 class)
Missing values: '?' in node-caps (8) and breast-quad (1)
"""


import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.profiling.data_profiler import DataProfiler

#set pandas display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def load_breast_cancer_dataset():
    df = pd.read_csv("datasets/breast+cancer/breast-cancer.csv")
    
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
