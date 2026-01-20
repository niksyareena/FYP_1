"""
Test duplicate detection on real Adult dataset
Run on sample first, then optionally on full dataset
"""

import pandas as pd
import time
from duplicate_detector import DuplicateDetector

def load_adult_data(use_sample=True, sample_size=1000):
    """Load the format-corrected Adult dataset"""
    print(f"\n{'='*70}")
    print("LOADING ADULT DATASET")
    print(f"{'='*70}\n")
    
    # Load the corrected dataset from format correction step
    df = pd.read_csv('data/output/adult_corrected.csv')
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"✓ Using format-corrected data from pipeline")
    
    if use_sample:
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"✓ Sampled {len(df_sample):,} rows for testing (set use_sample=False for full dataset)\n")
        return df_sample, len(df)
    else:
        print(f"✓ Using FULL dataset\n")
        return df, len(df)

def test_exact_duplicates(df):
    """Test exact duplicate detection"""
    print(f"\n{'='*70}")
    print("STEP 1: EXACT DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector()
    
    print("Detecting exact duplicates...")
    start_time = time.time()
    duplicate_rows = detector.detect_duplicates(df)
    elapsed_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Exact duplicates found: {len(duplicate_rows):,}")
    print(f"   Duplicate percentage: {len(duplicate_rows)/len(df)*100:.2f}%")
    print(f"   Unique rows: {len(df) - len(duplicate_rows):,}")
    print(f"   Execution time: {elapsed_time:.2f} seconds")
    
    if len(duplicate_rows) > 0:
        # Group duplicates together
        print(f"\n📋 Exact Duplicate Groups:")
        print(f"   Total duplicate groups found: {len(duplicate_rows)}")
        print(f"   Showing first 3 groups with FULL rows:\n")
        
        # Find which rows each duplicate matches
        shown = 0
        seen_duplicates = set()
        
        for dup_idx in duplicate_rows.index:
            if dup_idx in seen_duplicates or shown >= 3:
                break
                
            # Find all rows identical to this one
            dup_row = df.loc[dup_idx]
            matches = df[df.eq(dup_row).all(axis=1)].index.tolist()
            
            if len(matches) > 1:
                shown += 1
                seen_duplicates.update(matches)
                print(f"   {'='*65}")
                print(f"   GROUP {shown}: {len(matches)} identical rows")
                print(f"   Indices: {matches}")
                print(f"   {'='*65}")
                
                # Show FULL rows for all duplicates in this group
                for idx in matches:
                    print(f"\n   Row {idx} (full record):")
                    for col, val in df.loc[idx].items():
                        print(f"      {col}: {val}")
                print()  # Extra space between groups
    
    print(f"\n{'='*70}\n")
    return detector, duplicate_rows

def test_fuzzy_duplicates(df, threshold=0.8):
    """Test fuzzy duplicate detection"""
    print(f"\n{'='*70}")
    print(f"STEP 2: FUZZY DUPLICATE DETECTION (threshold={threshold})")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector(fuzzy_threshold=threshold)
    
    print(f"Detecting fuzzy duplicates...")
    print(f"Note: This uses exhaustive comparison (all pairs)")
    print(f"Expected comparisons: {len(df) * (len(df) - 1) // 2:,}\n")
    
    start_time = time.time()
    fuzzy_pairs = detector.detect_fuzzy_duplicates(df, threshold=threshold)
    elapsed_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Total rows checked: {len(df):,}")
    print(f"   Fuzzy duplicate pairs found: {len(fuzzy_pairs):,}")
    affected_rows = len(set([idx for pair in fuzzy_pairs for idx in pair[:2]]))
    print(f"   Affected rows: {affected_rows:,}")
    print(f"   Execution time: {elapsed_time:.2f} seconds")
    
    if len(fuzzy_pairs) > 0:
        print(f"\n📋 Fuzzy Duplicate Pairs (showing top 10 by similarity):\n")
        sorted_pairs = sorted(fuzzy_pairs, key=lambda x: x[2], reverse=True)[:10]
        
        for pair_num, (idx1, idx2, sim) in enumerate(sorted_pairs, 1):
            print(f"   {'='*65}")
            print(f"   Pair {pair_num}: Row {idx1} ↔ Row {idx2} (Similarity: {sim:.2%})")
            print(f"   {'='*65}")
            
            # Show key columns for comparison
            key_cols = ['age', 'workclass', 'education', 'occupation', 'marital-status', 'race']
            available_cols = [col for col in key_cols if col in df.columns]
            
            print(f"\n   Row {idx1}:")
            for col in available_cols:
                print(f"      {col}: {df.iloc[idx1][col]}")
            print(f"\n   Row {idx2}:")
            for col in available_cols:
                print(f"      {col}: {df.iloc[idx2][col]}")
            print()
        
        print(f"\n⚠️  IMPORTANT: Review these pairs carefully!")
        print(f"   Fuzzy matching may detect both:")
        print(f"   - True duplicates (same person with typos)")
        print(f"   - Similar records (different people with similar attributes)")
    else:
        print(f"\n✓ No fuzzy duplicates found at threshold {threshold}")
    
    print(f"\n{'='*70}\n")
    return detector, fuzzy_pairs

def main():
    """Run duplicate detection on Adult dataset"""
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION - ADULT DATASET")
    print(f"{'='*70}")
    
    overall_start = time.time()
    
    # Configuration
    USE_SAMPLE = False # Set to False to run on full dataset
    SAMPLE_SIZE = 1000  # Number of rows to sample
    THRESHOLD = 0.8
    
    # Load data
    df, total_rows = load_adult_data(use_sample=USE_SAMPLE, sample_size=SAMPLE_SIZE)
    
    # Step 1: Detect exact duplicates
    exact_detector, exact_duplicates = test_exact_duplicates(df)
    
    # Step 2: Detect fuzzy duplicates
    fuzzy_detector, fuzzy_pairs = test_fuzzy_duplicates(df, threshold=THRESHOLD)
    
    # Save results using detector's built-in method
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    fuzzy_detector.save_duplicates_log('data/output/adult_duplicate_detection_log.json')
    
    overall_time = time.time() - overall_start
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nDataset: Adult (format-corrected)")
    print(f"Total rows in full dataset: {total_rows:,}")
    print(f"Rows tested: {len(df):,} {'(SAMPLE)' if USE_SAMPLE else '(FULL)'}")
    print(f"\nResults:")
    print(f"   Exact duplicates: {len(exact_duplicates):,}")
    print(f"   Fuzzy duplicate pairs: {len(fuzzy_pairs):,} (threshold={THRESHOLD})")
    print(f"\n⏱️  Total Execution Time: {overall_time:.2f} seconds")
    
    if USE_SAMPLE:
        # Estimate full dataset time
        estimated_time = (overall_time / len(df)) * total_rows * (total_rows / len(df))
        print(f"\n💡 Estimated time for FULL dataset: ~{estimated_time/60:.1f} minutes")
        print(f"   To run on full dataset, set USE_SAMPLE = False in the script")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
