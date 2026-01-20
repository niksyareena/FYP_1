"""
Test Data Profiler with UCI Adult Dataset
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from src.profiling.data_profiler import DataProfiler


def load_adult_dataset():
    df = pd.read_csv("datasets/adult/adult.csv")
    
    return df


def main():
    print("\n" + "="*70)
    print("Testing Data Profiler with UCI Adult Dataset".center(70))
    print("="*70 + "\n")
    
    #set pandas display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    #load Adult dataset
    print("Loading UCI Adult dataset...")
    df = load_adult_dataset()
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    
    #show first few rows
    print("="*70)
    print("FIRST 5 ROWS".center(70))
    print("="*70)
    print(df.head())
    print()
    
    #create profiler
    profiler = DataProfiler()
    
    #generate profile
    print("\nGenerating comprehensive profile...\n")
    profile = profiler.generate_profile(df)
    
    #print summary
    profiler.print_summary()
    
    #save report
    print("\n" + "="*70)
    profiler.save_report('data/adult_profile_report.json')
    print("="*70)
    
    print("\n✓ Adult dataset profiling complete!\n")
    #if profile['numeric_columns']['columns']:
    #    numeric_stats = profile['numeric_columns']['columns']
    #    high_std = max(numeric_stats.items(), key=lambda x: x[1]['std'] if x[1]['std'] else 0)
    #    print(f"\nNumeric column with highest variability: {high_std[0]} (std={high_std[1]['std']:.2f})")
    
    print("\n✓ Adult dataset profiling complete!")


if __name__ == '__main__':
    main()
