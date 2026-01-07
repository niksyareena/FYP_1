"""
Test duplicate detection algorithm on Bank Marketing dataset
"""

import pandas as pd
from duplicate_detector import DuplicateDetector

def load_test_data():
    """Load synthetic test dataset"""
    df = pd.read_csv('data/output/test_duplicates_bank.csv')
    return df

def load_ground_truth():
    """Define ground truth duplicate pairs"""
    #updated for 500 base + 50 exact + 100 fuzzy
    n_base = 500
    n_exact = 50
    n_fuzzy = 100
    
    #exact duplicates: rows 0-49 copied to positions 500-549
    exact_pairs = [(i, n_base + i) for i in range(n_exact)]
    
    #fuzzy duplicates: rows 50-149 with variations at positions 550-649
    fuzzy_pairs = [(n_exact + i, n_base + n_exact + i) for i in range(n_fuzzy)]
    
    ground_truth = {
        'exact_pairs': exact_pairs,
        'fuzzy_pairs': fuzzy_pairs
    }
    
    return ground_truth

def test_exact_duplicate_detection(df, ground_truth):
    """Test exact duplicate detection"""
    print(f"\n{'='*70}")
    print("TEST 1: EXACT DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector()
    
    #detect exact duplicates (don't remove them)
    print("Running exact duplicate detection...")
    duplicates = detector.detect_duplicates(df)
    
    #get indices of detected duplicates (second occurrence when keep='first')
    detected_indices = set(duplicates.index.tolist())
    
    #convert ground truth to set of expected duplicate indices (second occurrence)
    expected_indices = set([idx2 for idx1, idx2 in ground_truth['exact_pairs']])
    
    #calculate metrics
    true_positives = len(detected_indices & expected_indices)
    false_positives = len(detected_indices - expected_indices)
    false_negatives = len(expected_indices - detected_indices)
    
    precision = true_positives / len(detected_indices) if len(detected_indices) > 0 else 0
    recall = true_positives / len(expected_indices) if len(expected_indices) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    #display results
    print(f"\n📊 Results:")
    print(f"   Expected duplicates: {len(expected_indices)}")
    print(f"   Detected duplicates: {len(detected_indices)}")
    print(f"   True Positives: {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    
    print(f"\n📈 Metrics:")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    
    if false_positives > 0:
        print(f"\n⚠️ False Positives (detected but not in ground truth):")
        for idx in sorted(detected_indices - expected_indices)[:5]:
            print(f"   Row {idx}")
    
    if false_negatives > 0:
        print(f"\n❌ False Negatives (missed):")
        for idx in sorted(expected_indices - detected_indices)[:5]:
            print(f"   Row {idx}")
    
    status = "✓ PASS" if precision == 1.0 and recall == 1.0 else "⚠ PARTIAL"
    print(f"\n{status}")
    print(f"{'='*70}\n")
    
    return detector, precision, recall, f1

def test_fuzzy_duplicate_detection(df, ground_truth, threshold=0.75):
    """Test fuzzy duplicate detection"""
    print(f"\n{'='*70}")
    print("TEST 2: FUZZY DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector(fuzzy_threshold=threshold)
    
    #detect fuzzy duplicates using exhaustive method
    print(f"Running fuzzy duplicate detection (threshold={threshold})...")
    fuzzy_pairs = detector.detect_fuzzy_duplicates(
        df, 
        threshold=threshold
    )
    
    #convert to set of tuples for comparison
    detected_pairs = {(min(idx1, idx2), max(idx1, idx2)) for idx1, idx2, sim in fuzzy_pairs}
    expected_pairs = {(min(idx1, idx2), max(idx1, idx2)) for idx1, idx2 in ground_truth['fuzzy_pairs']}
    
    #calculate metrics
    true_positives = len(detected_pairs & expected_pairs)
    false_positives = len(detected_pairs - expected_pairs)
    false_negatives = len(expected_pairs - detected_pairs)
    
    precision = true_positives / len(detected_pairs) if len(detected_pairs) > 0 else 0
    recall = true_positives / len(expected_pairs) if len(expected_pairs) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    #display results
    print(f"\n📊 Results:")
    print(f"   Expected fuzzy pairs: {len(expected_pairs)}")
    print(f"   Detected fuzzy pairs: {len(detected_pairs)}")
    print(f"   True Positives: {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    
    print(f"\n📈 Metrics:")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    
    if true_positives > 0:
        print(f"\n✓ Correctly Detected (sample of 5):")
        for idx1, idx2, sim in sorted(fuzzy_pairs, key=lambda x: x[2], reverse=True)[:5]:
            pair = (min(idx1, idx2), max(idx1, idx2))
            if pair in expected_pairs:
                print(f"   Row {idx1} ↔ Row {idx2} (similarity: {sim:.2%})")
    
    if false_positives > 0:
        print(f"\n⚠️ False Positives (detected but not in ground truth - sample of 5):")
        count = 0
        for idx1, idx2, sim in fuzzy_pairs:
            pair = (min(idx1, idx2), max(idx1, idx2))
            if pair not in expected_pairs:
                row1_data = df.iloc[idx1][['job', 'education', 'marital']].to_dict()
                row2_data = df.iloc[idx2][['job', 'education', 'marital']].to_dict()
                print(f"   Row {idx1} ↔ Row {idx2} (similarity: {sim:.2%})")
                print(f"     {row1_data}")
                print(f"     {row2_data}")
                count += 1
                if count >= 5:
                    break
    
    if false_negatives > 0:
        print(f"\n❌ False Negatives (missed from ground truth):")
        for pair in sorted(expected_pairs - detected_pairs)[:5]:
            idx1, idx2 = pair
            row1_data = df.iloc[idx1][['job', 'education', 'marital']].to_dict()
            row2_data = df.iloc[idx2][['job', 'education', 'marital']].to_dict()
            print(f"   Row {idx1} ↔ Row {idx2}")
            print(f"     {row1_data}")
            print(f"     {row2_data}")
    
    status = "✓ PASS" if precision >= 0.90 and recall >= 0.80 else "⚠ CHECK"
    print(f"\n{status}")
    print(f"{'='*70}\n")
    
    return detector, precision, recall, f1

def main():
    """Run all duplicate detection tests"""
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION ALGORITHM TEST - BANK MARKETING DATASET")
    print(f"{'='*70}\n")
    
    print("Loading synthetic test dataset...")
    df = load_test_data()
    print(f"✓ Loaded {len(df)} rows\n")
    
    ground_truth = load_ground_truth()
    print(f"Ground Truth Summary:")
    print(f"   Exact duplicate pairs: {len(ground_truth['exact_pairs'])}")
    print(f"   Fuzzy duplicate pairs: {len(ground_truth['fuzzy_pairs'])}")
    
    #test exact duplicate detection
    exact_detector, exact_precision, exact_recall, exact_f1 = test_exact_duplicate_detection(df, ground_truth)
    
    #test fuzzy duplicate detection
    fuzzy_detector, fuzzy_precision, fuzzy_recall, fuzzy_f1 = test_fuzzy_duplicate_detection(df, ground_truth, threshold=0.8)
    
    #final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nExact Duplicate Detection:")
    print(f"   Precision: {exact_precision:.2%}, Recall: {exact_recall:.2%}, F1: {exact_f1:.2%}")
    print(f"\nFuzzy Duplicate Detection (threshold=0.8):")
    print(f"   Precision: {fuzzy_precision:.2%}, Recall: {fuzzy_recall:.2%}, F1: {fuzzy_f1:.2%}")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
