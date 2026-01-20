"""
Test Format Correction Module with Bank Marketing Dataset
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from src.format_correction.format_corrector import FormatCorrector

#set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)


def load_bank_dataset():
    """Load Bank Marketing dataset"""
    df = pd.read_csv("datasets/bank-additional/bank-additional.csv")
    return df


def main():
    print("\n" + "="*70)
    print("FORMAT CORRECTION TEST - BANK MARKETING DATASET".center(70))
    print("="*70 + "\n")
    
    #load dataset
    print("Loading Bank Marketing dataset...")
    df = load_bank_dataset()
    print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    
    #show sample before correction
    print("="*70)
    print("SAMPLE DATA (BEFORE CORRECTION)".center(70))
    print("="*70)
    print(df.head())
    print()
    
    #create format corrector
    corrector = FormatCorrector()
    
    #configure corrections
    config = {
        'normalize_strings': True,
        'standardize_dates': False,  #Bank dataset has no date columns
        'correct_types': True,
        'string_case': 'lower'
    }
    
    print("="*70)
    print("APPLYING FORMAT CORRECTIONS".center(70))
    print()
    
    #apply corrections
    df_corrected = corrector.correct_formats(df, config)
    
    #print summary
    corrector.print_summary()
    
    #show sample after correction
    print("\n" + "="*70)
    print("SAMPLE DATA (AFTER CORRECTION)".center(70))
    print("="*70)
    print(df_corrected.head())
    print()
    
    #show before/after comparison for corrected columns only
    print("="*70)
    print("BEFORE/AFTER COMPARISON (Sample Rows)".center(70))
    print("="*70)
    
    #get list of corrected columns from log (exclude 'all_columns' meta entry, deduplicate)
    corrected_cols = list(dict.fromkeys([item['column'] for item in corrector.corrections_log if item['column'] != 'all_columns']))
    
    if not corrected_cols:
        print("\n   No corrections were applied.\n")
    else:
        for col in corrected_cols:
            print(f"\n📋 Column: {col}")
            print(f"   {'Before':<40} {'After':<40}")
            print(f"   {'-'*40} {'-'*40}")
            for i in range(min(5, len(df))):
                before = str(df[col].iloc[i])[:38]
                after = str(df_corrected[col].iloc[i])[:38]
                print(f"   {before:<40} {after:<40}")
    
    #save corrected dataset
    print("\n" + "="*70)
    output_path = 'data/output/bank_corrected.csv'
    df_corrected.to_csv(output_path, index=False)
    print(f"✓ Corrected dataset saved to: {output_path}")
    
    #save corrections log
    corrector.save_corrections_log('data/output/bank_format_corrections.json')
    print("="*70)
    
    print("\n✓ Format correction test complete!\n")


if __name__ == '__main__':
    main()
