"""
Test duplicate detection on real Bank Marketing dataset
Run on sample first, then optionally on full dataset
"""

import pandas as pd
import time
from duplicate_detector import DuplicateDetector

def load_bank_data():
    """Load the format-corrected Bank Marketing dataset"""
    print(f"\n{'='*70}")
    print("LOADING BANK MARKETING DATASET")
    print(f"{'='*70}\n")
    
    # Load the corrected dataset from format correction step
    df = pd.read_csv('data/output/bank_corrected.csv')
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"✓ Using format-corrected data from pipeline\n")
    
    return df

def test_exact_duplicates(df, detector):
    """Test exact duplicate detection"""
    print(f"\n{'='*70}")
    print("STEP 1: EXACT DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
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
        #group duplicates together
        print(f"\n📋 Exact Duplicate Groups:")
        print(f"   Total duplicate groups found: {len(duplicate_rows)}")
        print(f"   Showing first 3 groups:\n")
        
        #find which rows each duplicate matches
        shown = 0
        seen_duplicates = set()
        
        for dup_idx in duplicate_rows.index:
            if dup_idx in seen_duplicates or shown >= 3:
                break
                
            # find all rows identical to this one
            dup_row = df.loc[dup_idx]
            matches = df[df.eq(dup_row).all(axis=1)].index.tolist()
            
            if len(matches) > 1:
                shown += 1
                seen_duplicates.update(matches)
                print(f"   {'='*65}")
                print(f"   GROUP {shown}: {len(matches)} identical rows")
                print(f"   Indices: {matches}")
                print(f"   {'='*65}")
                
                #show full rows
                for idx in matches:
                    print(f"\n   Row {idx} (full record):")
                    for col, val in df.loc[idx].items():
                        print(f"      {col}: {val}")
                print()  
    
    print(f"\n{'='*70}\n")
    return duplicate_rows

def test_fuzzy_duplicates(df, detector):
    """Test fuzzy duplicate detection"""
    print(f"\n{'='*70}")
    print(f"STEP 2: FUZZY DUPLICATE DETECTION (threshold=0.80)")
    print(f"{'='*70}\n")
    
    print(f"Detecting fuzzy duplicates...")
    print(f"Note: This uses exhaustive comparison (all pairs)")
    print(f"Expected comparisons: {len(df) * (len(df) - 1) // 2:,}\n")
    
    start_time = time.time()
    fuzzy_pairs = detector.detect_fuzzy_duplicates(df)
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
            
            #show key columns for comparison
            key_cols = ['age', 'job', 'marital', 'education', 'housing', 'loan']
            available_cols = [col for col in key_cols if col in df.columns]
            
            print(f"\n   Row {idx1}:")
            for col in available_cols:
                print(f"      {col}: {df.iloc[idx1][col]}")
            print(f"\n   Row {idx2}:")
            for col in available_cols:
                print(f"      {col}: {df.iloc[idx2][col]}")
            print()
        
        # print(f"\n⚠️  IMPORTANT: Review these pairs carefully!")
        # print(f"   Fuzzy matching may detect both:")
        # print(f"   - True duplicates (same person with typos)")
        # print(f"   - Similar records (different people with similar attributes)")
    else:
        print(f"\n✓ No fuzzy duplicates found at threshold 0.80")
    
    print(f"\n{'='*70}\n")
    return detector, fuzzy_pairs

def main():
    """Run duplicate detection on Bank Marketing dataset"""
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION - BANK MARKETING DATASET")
    print(f"{'='*70}")
    
    overall_start = time.time()
    
    #load data
    df = load_bank_data()
    
    #create single detector instance for both operations
    detector = DuplicateDetector()
    
    #detect exact duplicates
    exact_duplicates = test_exact_duplicates(df, detector)
    
    #detect fuzzy duplicates
    fuzzy_pairs = test_fuzzy_duplicates(df, detector)

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    detector.save_duplicates_log('data/output/bank_duplicate_detection_log.json')
    
    overall_time = time.time() - overall_start
    
    #final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nDataset: Bank Marketing (format-corrected)")
    print(f"Total rows tested: {len(df):,}")
    print(f"\nResults:")
    print(f"   Exact duplicates: {len(exact_duplicates):,}")
    print(f"   Fuzzy duplicate pairs: {len(fuzzy_pairs):,} (threshold=0.80)")
    print(f"\n⏱️  Total Execution Time: {overall_time:.2f} seconds")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
