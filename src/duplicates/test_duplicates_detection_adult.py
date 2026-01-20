"""
Test duplicate detection algorithm on synthetic dataset with injected duplicates
"""

import pandas as pd
import time
from duplicate_detector import DuplicateDetector

def load_test_data():
    print("Loading synthetic test dataset...")
    df = pd.read_csv('data/output/test_duplicates_injected_adult.csv')
    print(f"✓ Loaded {len(df)} rows\n")
    return df

def load_ground_truth():
    #500 base + 5 exact (integration check) + 100 fuzzy (real testing)
    n_base = 500
    n_exact = 5
    n_fuzzy = 100
    
    #exact duplicates: rows 0-4 copied to positions 500-504
    exact_pairs = [(i, n_base + i) for i in range(n_exact)]
    
    #fuzzy duplicates: rows 5-104 with variations at positions 505-604
    fuzzy_pairs = [(n_exact + i, n_base + n_exact + i) for i in range(n_fuzzy)]
    
    ground_truth = {
        'exact_pairs': exact_pairs,
        'fuzzy_pairs': fuzzy_pairs
    }
    
    return ground_truth

def test_exact_duplicate_detection(df, ground_truth):
    """test exact duplicate detection"""
    print(f"\n{'='*70}")
    print("TEST 1: EXACT DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector()
    
    #detect duplicates
    start_time = time.time()
    duplicate_rows = detector.detect_duplicates(df)
    execution_time = time.time() - start_time
    print(f"✓ Exact duplicate detection completed in {execution_time:.3f} seconds\n")
    
    #print detector summary
    #detector.print_summary()
    
    #get detected pairs
    detected_indices = set(duplicate_rows.index.tolist()) if duplicate_rows is not None else set()
    
    #compare with ground truth
    expected_pairs = set(ground_truth['exact_pairs'])
    expected_duplicate_indices = {idx2 for idx1, idx2 in expected_pairs}
    
    #calculate metrics
    true_positives = detected_indices & expected_duplicate_indices
    false_positives = detected_indices - expected_duplicate_indices
    false_negatives = expected_duplicate_indices - detected_indices
    
    precision = len(true_positives) / len(detected_indices) if detected_indices else 0
    recall = len(true_positives) / len(expected_duplicate_indices) if expected_duplicate_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    #print results
    print(f"\n📊 Results:")
    print(f"   Expected duplicates: {len(expected_duplicate_indices)}")
    print(f"   Detected duplicates: {len(detected_indices)}")
    print(f"   True Positives: {len(true_positives)}")
    print(f"   False Positives: {len(false_positives)}")
    print(f"   False Negatives: {len(false_negatives)}")
    
    print(f"\n📈 Metrics:")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    
    if false_positives:
        print(f"\n⚠️ False Positives (detected but not in ground truth):")
        for idx in sorted(false_positives):
            print(f"   Row {idx}")
    
    if false_negatives:
        print(f"\n❌ False Negatives (missed):")
        for idx in sorted(false_negatives):
            print(f"   Row {idx}")
    
    #status = "✓ PASS" if precision == 1.0 and recall == 1.0 else "⚠ PARTIAL"
    #print(f"\n{status}")
    print(f"{'='*70}\n")
    
    return detector, precision, recall, f1

def test_fuzzy_duplicate_detection(df, ground_truth, threshold=0.75):
    """test fuzzy duplicate detection"""
    print(f"\n{'='*70}")
    print("TEST 2: FUZZY DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector(fuzzy_threshold=threshold)
    
    #detect fuzzy duplicates using exhaustive method
    start_time = time.time()
    fuzzy_pairs = detector.detect_fuzzy_duplicates(
        df,
        threshold=threshold
    )
    execution_time = time.time() - start_time
    print(f"\n⏱️  Fuzzy detection execution time: {execution_time:.3f} seconds\n")
    
    #convert to set of tuples for comparison
    detected_pairs = {(min(idx1, idx2), max(idx1, idx2)) for idx1, idx2, sim in fuzzy_pairs}
    expected_pairs = {(min(idx1, idx2), max(idx1, idx2)) for idx1, idx2 in ground_truth['fuzzy_pairs']}
    
    #calculate metrics
    true_positives = detected_pairs & expected_pairs
    false_positives = detected_pairs - expected_pairs
    false_negatives = expected_pairs - detected_pairs
    
    precision = len(true_positives) / len(detected_pairs) if detected_pairs else 0
    recall = len(true_positives) / len(expected_pairs) if expected_pairs else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    #print results
    print(f"\n📊 Results:")
    print(f"   Expected fuzzy pairs: {len(expected_pairs)}")
    print(f"   Detected fuzzy pairs: {len(detected_pairs)}")
    print(f"   True Positives: {len(true_positives)}")
    print(f"   False Positives: {len(false_positives)}")
    print(f"   False Negatives: {len(false_negatives)}")
    
    print(f"\n📈 Metrics:")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    
    if true_positives:
        print(f"\n✓ Correctly Detected Fuzzy Duplicates (sample of 5):")
        for idx1, idx2 in sorted(true_positives)[:5]:
            sim = next((s for i1, i2, s in fuzzy_pairs if (min(i1, i2), max(i1, i2)) == (idx1, idx2)), 0)
            print(f"\n   Pair: Row {idx1} ↔ Row {idx2} (similarity: {sim:.2%})")
            
            # Show key columns for comparison
            cols_to_show = ['workclass', 'education', 'occupation', 'marital-status']
            row1_data = df.iloc[idx1][cols_to_show]
            row2_data = df.iloc[idx2][cols_to_show]
            
            print(f"   Row {idx1}:")
            for col in cols_to_show:
                print(f"      {col}: {row1_data[col]}")
            print(f"   Row {idx2}:")
            for col in cols_to_show:
                print(f"      {col}: {row2_data[col]}")
    
    if false_positives:
        print(f"\n⚠️ False Positives (detected but not in ground truth - sample of 5):")
        for idx1, idx2 in sorted(false_positives)[:5]:
            sim = next((s for i1, i2, s in fuzzy_pairs if (min(i1, i2), max(i1, i2)) == (idx1, idx2)), 0)
            print(f"   Row {idx1} ↔ Row {idx2} (similarity: {sim:.2%})")
            row1 = df.iloc[idx1][['workclass', 'education', 'occupation']].to_dict()
            row2 = df.iloc[idx2][['workclass', 'education', 'occupation']].to_dict()
            print(f"     {row1}")
            print(f"     {row2}")
    
    if false_negatives:
        print(f"\n❌ False Negatives (missed from ground truth):")
        for idx1, idx2 in sorted(false_negatives):
            row1 = df.iloc[idx1][['workclass', 'education', 'occupation']].to_dict()
            row2 = df.iloc[idx2][['workclass', 'education', 'occupation']].to_dict()
            print(f"   Row {idx1} ↔ Row {idx2}")
            print(f"     {row1}")
            print(f"     {row2}")
    
    #status = "✓ PASS" if precision >= 0.8 and recall >= 0.8 else "⚠ CHECK"
    #print(f"\n{status}")
    #print(f"{'='*70}\n")
    
    return detector, precision, recall, f1

def main():
    """run all detection tests"""
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION ALGORITHM TEST")
    print(f"{'='*70}\n")
    
    overall_start = time.time()
    
    #load data
    df = load_test_data()
    ground_truth = load_ground_truth()
    
    print(f"Ground Truth Summary:")
    print(f"   Exact duplicate pairs: {len(ground_truth['exact_pairs'])}")
    print(f"   Fuzzy duplicate pairs: {len(ground_truth['fuzzy_pairs'])}")
    
    #test exact duplicate detection
    exact_detector, exact_precision, exact_recall, exact_f1 = test_exact_duplicate_detection(df, ground_truth)
    
    #test fuzzy duplicate detection 
    fuzzy_detector, fuzzy_precision, fuzzy_recall, fuzzy_f1 = test_fuzzy_duplicate_detection(df, ground_truth, threshold=0.8)
    
    overall_time = time.time() - overall_start
    
    #final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nExact Duplicate Detection:")
    print(f"   Precision: {exact_precision:.2%}, Recall: {exact_recall:.2%}, F1: {exact_f1:.2%}")
    print(f"\nFuzzy Duplicate Detection (threshold=0.8):")
    print(f"   Precision: {fuzzy_precision:.2%}, Recall: {fuzzy_recall:.2%}, F1: {fuzzy_f1:.2%}")
    print(f"\n⏱️  Total Execution Time: {overall_time:.3f} seconds")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
