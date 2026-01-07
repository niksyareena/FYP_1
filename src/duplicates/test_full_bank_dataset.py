"""
Test duplicate detection on full Bank Marketing dataset (4,119 rows)
"""

import pandas as pd
import sys
import os
sys.path.insert(0, 'src')
from duplicates.duplicate_detector import DuplicateDetector
from format_correction.format_corrector import FormatCorrector

def load_full_bank_dataset():
    """Load and prepare full Bank Marketing dataset"""
    print(f"\n{'='*70}")
    print("LOADING FULL BANK MARKETING DATASET")
    print(f"{'='*70}\n")
    
    #load dataset
    df = pd.read_csv('datasets/bank-additional/bank-additional.csv', sep=';')
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"✓ Dataset shape: {df.shape}")
    
    return df

def test_exact_duplicates(df):
    """Test exact duplicate detection"""
    print(f"\n{'='*70}")
    print("STEP 1: EXACT DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector()
    
    print("Detecting exact duplicates...")
    duplicates = detector.detect_duplicates(df)
    
    print(f"\n📊 Results:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Exact duplicates found: {len(duplicates):,}")
    print(f"   Duplicate percentage: {len(duplicates)/len(df)*100:.2f}%")
    print(f"   Unique rows: {len(df) - len(duplicates):,}")
    
    if len(duplicates) > 0:
        print(f"\n📋 Sample Duplicates (first 5):")
        for idx in duplicates.index[:5]:
            print(f"   Row {idx}: {dict(df.loc[idx][['job', 'education', 'marital', 'contact']].head())}")
    
    return detector, duplicates

def test_fuzzy_duplicates(df, threshold=0.85):
    """Test fuzzy duplicate detection on full dataset with blocking"""
    print(f"\n{'='*70}")
    print(f"STEP 2: FUZZY DUPLICATE DETECTION (threshold={threshold})")
    print(f"{'='*70}\n")
    
    #apply format correction first
    print("Applying format correction...")
    corrector = FormatCorrector()
    df_corrected = corrector.normalize_strings(df, case='lower', normalize_punctuation=True)
    print("✓ Format correction applied\n")
    
    detector = DuplicateDetector(fuzzy_threshold=threshold)
    
    #automatically detect string columns for fuzzy matching
    string_columns = df_corrected.select_dtypes(include=['object']).columns.tolist()
    #exclude long text columns or IDs if present
    key_columns = [col for col in string_columns if df_corrected[col].str.len().mean() < 50]
    
    print(f"Auto-detected key columns for fuzzy matching: {key_columns}")
    
    #use blocking strategy for performance
    blocking_key = key_columns[0] if key_columns else None
    
    if blocking_key:
        fuzzy_pairs = detector.detect_fuzzy_duplicates_with_blocking(
            df_corrected,
            blocking_key=blocking_key,
            subset=key_columns,
            threshold=threshold
        )
    else:
        fuzzy_pairs = detector.detect_fuzzy_duplicates(
            df_corrected,
            subset=key_columns,
            threshold=threshold
        )
    
    print(f"\n📊 Results:")
    print(f"   Total rows checked: {len(df_corrected):,}")
    print(f"   Fuzzy duplicate pairs found: {len(fuzzy_pairs):,}")
    print(f"   Affected rows: {len(set([idx for pair in fuzzy_pairs for idx in pair[:2]])):,}")
    
    if len(fuzzy_pairs) > 0:
        print(f"\n📋 Sample Fuzzy Duplicate Pairs (first 10 by similarity):")
        for idx1, idx2, sim in sorted(fuzzy_pairs, key=lambda x: x[2], reverse=True)[:10]:
            row1 = df_corrected.iloc[idx1][key_columns]
            row2 = df_corrected.iloc[idx2][key_columns]
            print(f"\n   Row {idx1} ↔ Row {idx2} (similarity: {sim:.2%})")
            print(f"     Row {idx1}: {dict(row1)}")
            print(f"     Row {idx2}: {dict(row2)}")
        
        print(f"\n⚠️  WARNING: Fuzzy duplicates detected on demographic data without unique identifiers.")
        print(f"   These pairs may include false positives (similar but distinct individuals).")
        print(f"   RECOMMENDATION: Review pairs manually before removal or use supervised approach.")
    
    return detector, fuzzy_pairs, key_columns

def test_combined_cleaning(df, threshold=0.85):
    """Test combined exact + fuzzy duplicate removal with user review"""
    print(f"\n{'='*70}")
    print("STEP 3: COMBINED DUPLICATE REMOVAL WITH REVIEW")
    print(f"{'='*70}\n")
    
    initial_count = len(df)
    
    #apply format correction
    corrector = FormatCorrector()
    df_clean = corrector.normalize_strings(df, case='lower', normalize_punctuation=True)
    
    #remove exact duplicates
    detector = DuplicateDetector()
    duplicates = detector.detect_duplicates(df_clean)
    df_clean = detector.remove_duplicates(df_clean, keep='first')
    exact_removed = len(duplicates)
    
    print(f"✓ Removed {exact_removed:,} exact duplicates (auto-removed)")
    print(f"   Rows remaining: {len(df_clean):,}")
    
    #auto-detect string columns
    string_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    key_columns = [col for col in string_columns if df_clean[col].str.len().mean() < 50]
    blocking_key = key_columns[0] if key_columns else None
    
    print(f"\nDetecting fuzzy duplicates with blocking strategy...")
    
    if blocking_key:
        fuzzy_pairs = detector.detect_fuzzy_duplicates_with_blocking(
            df_clean,
            blocking_key=blocking_key,
            subset=key_columns,
            threshold=threshold
        )
    else:
        fuzzy_pairs = detector.detect_fuzzy_duplicates(
            df_clean,
            subset=key_columns,
            threshold=threshold
        )
    
    #prompt for review
    if len(fuzzy_pairs) > 0:
        #use interactive removal with user review
        df_clean = detector.remove_fuzzy_duplicates(df_clean, fuzzy_pairs=fuzzy_pairs, keep='first', interactive=True)
        fuzzy_removed = len(df_clean) - (initial_count - exact_removed)
    else:
        fuzzy_removed = 0
    
    print(f"✓ Removed {fuzzy_removed:,} fuzzy duplicates")
    print(f"   Rows remaining: {len(df_clean):,}")
    
    print(f"\n📊 Final Summary:")
    print(f"   Original rows: {initial_count:,}")
    print(f"   Exact duplicates removed: {exact_removed:,}")
    print(f"   Fuzzy duplicates removed: {abs(fuzzy_removed):,}")
    print(f"   Total removed: {exact_removed + abs(fuzzy_removed):,}")
    print(f"   Final clean rows: {len(df_clean):,}")
    print(f"   Data reduction: {(exact_removed + abs(fuzzy_removed))/initial_count*100:.2f}%")
    
    #save cleaned dataset
    output_path = 'data/output/bank_cleaned_full.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Saved cleaned dataset to '{output_path}'")
    
    return df_clean

def main():
    """Run full dataset duplicate detection tests"""
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION TEST - FULL BANK MARKETING DATASET")
    print(f"{'='*70}")
    
    #load full dataset
    df = load_full_bank_dataset()
    
    #test 1: exact duplicates
    detector, duplicates = test_exact_duplicates(df)
    
    #test fuzzy duplicates (with threshold=0.85 and blocking)
    fuzzy_detector, fuzzy_pairs, key_columns = test_fuzzy_duplicates(df, threshold=0.85)
    
    #test 3: combined cleaning
    df_clean = test_combined_cleaning(df, threshold=0.85)
    
    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
