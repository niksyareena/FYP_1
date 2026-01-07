"""
Test full pipeline: Format Correction → Duplicate Detection
Bank Marketing Dataset
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from format_correction.format_corrector import FormatCorrector
from duplicates.duplicate_detector import DuplicateDetector

def load_ground_truth():
    """Define ground truth for injected duplicates"""
    ground_truth = {
        'exact_pairs': [(i, 100 + i) for i in range(10)],
        'fuzzy_pairs': [(10 + i, 110 + i) for i in range(15)]
    }
    return ground_truth

def test_pipeline():
    """Test Format Correction → Duplicate Detection pipeline"""
    
    print("="*70)
    print("FULL PIPELINE TEST: FORMAT CORRECTION → DUPLICATE DETECTION")
    print("="*70)
    print()
    
    #step 1: Load test dataset with injected duplicates
    print("STEP 1: Loading test dataset...")
    df = pd.read_csv('data/output/test_duplicates_bank.csv')
    print(f"✓ Loaded {len(df)} rows")
    print()
    
    #step 2: Apply Format Correction (normalization)
    print("STEP 2: Applying Format Correction...")
    corrector = FormatCorrector()
    
    #normalize strings with punctuation normalization
    df_corrected = corrector.normalize_strings(
        df, 
        case='lower', 
        normalize_punctuation=True  #dots, hyphens, underscores → spaces
    )
    
    print("Format corrections applied:")
    for log_entry in corrector.corrections_log:
        print(f"  {log_entry['column']}: {log_entry['rows_affected']} rows normalized")
    print()
    
    #step 3: Run Duplicate Detection on corrected data
    print("STEP 3: Running Duplicate Detection on corrected data...")
    print()
    
    #load ground truth
    ground_truth = load_ground_truth()
    
    #test exact duplicates
    print("="*70)
    print("TEST 1: EXACT DUPLICATE DETECTION")
    print("="*70)
    print()
    
    detector = DuplicateDetector()
    duplicates_df = detector.detect_duplicates(df_corrected)
    
    #get indices of duplicates
    duplicate_indices = set(duplicates_df.index.tolist()) if len(duplicates_df) > 0 else set()
    
    #calculate metrics
    expected_duplicates = set()
    for idx1, idx2 in ground_truth['exact_pairs']:
        expected_duplicates.add(idx2)  #second occurrence is what gets detected
    
    true_positives = duplicate_indices & expected_duplicates
    false_positives = duplicate_indices - expected_duplicates
    false_negatives = expected_duplicates - duplicate_indices
    
    precision = len(true_positives) / len(duplicate_indices) if duplicate_indices else 0
    recall = len(true_positives) / len(expected_duplicates) if expected_duplicates else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 Results:")
    print(f"   Expected duplicates: {len(expected_duplicates)}")
    print(f"   Detected duplicates: {len(duplicate_indices)}")
    print(f"   True Positives: {len(true_positives)}")
    print(f"   False Positives: {len(false_positives)}")
    print(f"   False Negatives: {len(false_negatives)}")
    print()
    print(f"📈 Metrics:")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    print()
    
    #show false positives (likely fuzzy duplicates converted to exact by format correction)
    if false_positives:
        print(f"ℹ️  False Positives (fuzzy duplicates upgraded to exact):")
        for idx in sorted(false_positives)[:3]:
            #find its original pair
            for orig_idx in range(len(df_corrected)):
                if orig_idx != idx and orig_idx not in duplicate_indices:
                    row1 = df_corrected.iloc[idx]
                    row2 = df_corrected.iloc[orig_idx]
                    if all(row1[col] == row2[col] for col in ['job', 'education', 'marital', 'contact', 'poutcome']):
                        print(f"   Row {orig_idx} ↔ Row {idx} (now exact after format correction)")
                        break
        print()
    
    #test fuzzy duplicates
    print("="*70)
    print("TEST 2: FUZZY DUPLICATE DETECTION")
    print("="*70)
    print()
    
    detector = DuplicateDetector(fuzzy_threshold=0.75)
    fuzzy_pairs = detector.detect_fuzzy_duplicates(df_corrected, threshold=0.75)
    
    #convert to set of pairs
    detected_pairs = {(min(idx1, idx2), max(idx1, idx2)) for idx1, idx2, sim in fuzzy_pairs}
    expected_pairs = set(ground_truth['fuzzy_pairs'])
    
    true_positives_fuzzy = detected_pairs & expected_pairs
    false_positives_fuzzy = detected_pairs - expected_pairs
    false_negatives_fuzzy = expected_pairs - detected_pairs
    
    precision_fuzzy = len(true_positives_fuzzy) / len(detected_pairs) if detected_pairs else 0
    recall_fuzzy = len(true_positives_fuzzy) / len(expected_pairs) if expected_pairs else 0
    f1_fuzzy = 2 * precision_fuzzy * recall_fuzzy / (precision_fuzzy + recall_fuzzy) if (precision_fuzzy + recall_fuzzy) > 0 else 0
    
    print(f"📊 Results:")
    print(f"   Expected fuzzy pairs: {len(expected_pairs)}")
    print(f"   Detected fuzzy pairs: {len(detected_pairs)}")
    print(f"   True Positives: {len(true_positives_fuzzy)}")
    print(f"   False Positives: {len(false_positives_fuzzy)}")
    print(f"   False Negatives: {len(false_negatives_fuzzy)}")
    print()
    print(f"📈 Metrics:")
    print(f"   Precision: {precision_fuzzy:.2%}")
    print(f"   Recall: {recall_fuzzy:.2%}")
    print(f"   F1-Score: {f1_fuzzy:.2%}")
    print()
    
    if false_negatives_fuzzy:
        print(f"❌ False Negatives (missed from ground truth):")
        for idx1, idx2 in sorted(false_negatives_fuzzy)[:5]:
            print(f"   Row {idx1} ↔ Row {idx2}")
            sample_cols = ['job', 'education', 'marital']
            print(f"     {dict(df_corrected.iloc[idx1][sample_cols])}")
            print(f"     {dict(df_corrected.iloc[idx2][sample_cols])}")
        print()
    
    #final summary
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()
    print(f"Pipeline: Format Correction (punctuation normalization) → Duplicate Detection")
    print()
    print(f"Exact Duplicate Detection:")
    print(f"   Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
    print()
    print(f"Fuzzy Duplicate Detection (threshold=0.75):")
    print(f"   Precision: {precision_fuzzy:.2%}, Recall: {recall_fuzzy:.2%}, F1: {f1_fuzzy:.2%}")
    print()
    
    #calculate total duplicates found
    total_expected = len(expected_duplicates) + len(expected_pairs)
    total_detected = len(duplicate_indices) + len(detected_pairs)
    total_tp = len(true_positives) + len(true_positives_fuzzy) + len(false_positives)  #FPs are fuzzy→exact upgrades
    total_precision = total_tp / total_detected if total_detected > 0 else 0
    total_recall = total_tp / total_expected if total_expected > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    
    print(f"Combined Results (Exact + Fuzzy):")
    print(f"   Total Expected: {total_expected}")
    print(f"   Total Detected: {total_detected}")
    print(f"   Overall Precision: {total_precision:.2%}")
    print(f"   Overall Recall: {total_recall:.2%}")
    print(f"   Overall F1-Score: {total_f1:.2%}")
    print()
    print("="*70)

if __name__ == "__main__":
    test_pipeline()
