"""
Duplicate Detector Module
Rule-based detection and removal of exact and fuzzy duplicate rows
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Literal
import json
import re
import Levenshtein


class DuplicateDetector:
    """
    Rule-based duplicate detection for exact and fuzzy row matches.
    
    Operations:
    - Detect exact duplicate rows
    - Detect fuzzy duplicate rows (similar but not identical)
    - Remove duplicates (keep first/last occurrence)
    - Log all duplicate removals
    """
    
    def __init__(self, fuzzy_threshold: float = 0.80):
        """
        Initialize duplicate detector
        
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0 to 1.0)
                           Default 0.80 means each field in a row pair must be 80% similar to be considered fuzzy duplicates
        """
        self.duplicates_log: List[Dict[str, Any]] = []
        self.original_count: int = 0
        self.duplicates_removed: int = 0
        self.exact_duplicates_removed: int = 0
        self.fuzzy_duplicates_removed: int = 0
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_duplicate_pairs: List[Tuple[int, int, float]] = []  #(idx1, idx2, similarity)
    
    def is_typo_match(self, str1: str, str2: str, threshold: float) -> Tuple[float, bool]:
        """
        Check if two strings are similar due to typos (not genuine value differences).
        
        Examples:
            - 'john' vs 'jonh' -> (0.88, True)   # Typo in name
            - 'basic.9y' vs 'basik.9y' -> (0.89, True)   # Typo, same education
            - 'basic.9y' vs 'basic.4y' -> (0.0, False)   # Different education levels
        
        Args:
            str1: First string to compare
            str2: Second string to compare  
            threshold: Minimum similarity to consider a typo match
            
        Returns:
            Tuple of (similarity_score, is_typo_match)
            - similarity_score: 0.0 to 1.0 (0.0 if numbers differ)
            - is_typo_match: True if similarity >= threshold AND numbers match
        """
        str1_clean = str(str1).lower().strip()
        str2_clean = str(str2).lower().strip()
        
        #exact match
        if str1_clean == str2_clean:
            return 1.0, True
        
        #extract numeric parts from both strings
        nums1 = re.findall(r'\d+', str1_clean)
        nums2 = re.findall(r'\d+', str2_clean)
        
        #if either string contains numbers and they differ, not a typo
        #typos affect text characters, not numbers
        if (nums1 or nums2) and nums1 != nums2:
            return 0.0, False
        
        #numbers match (or neither has numbers) - use levenshtein for typo detection
        similarity = Levenshtein.ratio(str1_clean, str2_clean)
        return similarity, similarity >= threshold
    
    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None):
        """
        Detect exact duplicate rows in the dataframe
        
        Args:
            df: Input dataframe
            subset: List of columns to consider for duplicates (None = all columns)
            
        Returns:
            DataFrame containing only the duplicate rows (excluding first occurrence)
        """
        #find duplicates (marks all duplicates except first occurrence)
        duplicate_mask = df.duplicated(subset=subset, keep='first')
        duplicate_rows = df[duplicate_mask].copy()  #type: ignore
        duplicates_found = len(duplicate_rows)
        
        #log detection
        self.duplicates_log.append({
            'operation': 'exact_detection',
            'columns_checked': subset if subset else 'all',
            'total_rows': len(df),
            'duplicates_found': duplicates_found,
            'duplicate_percentage': round(duplicates_found / len(df) * 100, 2) if len(df) > 0 else 0
        })
        
        return duplicate_rows
    
    def calc_row_similarity_per_field(self, row1: pd.Series, row2: pd.Series, 
                                   per_field_threshold: float) -> Tuple[float, bool, List[Tuple[str, float]]]:
        """
        Calculate similarity between two rows with per-field threshold checking
        
        Args:
            row1: First row
            row2: Second row
            per_field_threshold: Minimum similarity required for EACH field individually
            
        Returns:
            Tuple of (average_similarity, all_fields_pass, field_details)
            - average_similarity: Overall similarity score between 0.0 and 1.0
            - all_fields_pass: True only if ALL fields meet the threshold
            - field_details: List of (column_name, similarity_score) for each field
        """
        if len(row1) != len(row2):
            return 0.0, False, []
        
        field_similarities = []
        all_pass = True
        
        for col_name, val1, val2 in zip(row1.index, row1, row2):
            #handle missing values
            if pd.isna(val1) and pd.isna(val2):
                field_similarities.append((col_name, 1.0))
                continue
            elif pd.isna(val1) or pd.isna(val2):
                field_similarities.append((col_name, 0.0))
                all_pass = False
                break  #early stop: field failed, no point checking rest
            
            #convert to strings for comparison
            str1 = str(val1).lower().strip()
            str2 = str(val2).lower().strip()
            
            #calculate similarity using typo-aware matching
            #detects typos but rejects different numeric values
            similarity, is_typo = self.is_typo_match(str1, str2, per_field_threshold)
            
            field_similarities.append((col_name, similarity))
            
            #if numbers differ, this field fails immediately (not a typo)
            if not is_typo and similarity == 0.0:
                all_pass = False
                break
            
            #early stopping: if this field fails threshold, skip remaining fields
            if similarity < per_field_threshold:
                all_pass = False
                break  #no need to check remaining fields
        
        #calculate average similarity (only for fields checked)
        avg_similarity = sum(sim for _, sim in field_similarities) / len(field_similarities) if field_similarities else 0.0
        
        return avg_similarity, all_pass, field_similarities
    
    def detect_fuzzy_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None,
                               threshold: Optional[float] = None) -> List[Tuple[int, int, float]]:
        """
        Detect fuzzy duplicate rows in the dataframe
        
        Args:
            df: Input dataframe
            subset: List of columns to consider for duplicates (None = all columns)
            threshold: Similarity threshold (overrides instance threshold if provided)
            
        Returns:
            List of tuples (index1, index2, similarity_score) for detected fuzzy duplicates
        """
        threshold = threshold if threshold is not None else self.fuzzy_threshold
        
        fuzzy_pairs = []
        cols_to_check = subset if subset else df.columns.tolist()
        
        #compare each pair of rows
        for i in range(len(df)):
            # Show progress at 25%, 50%, 75% only
            if i > 0 and len(df) >= 100:
                if i == len(df) // 4:
                    print(f"  Progress: 25% complete...")
                elif i == len(df) // 2:
                    print(f"  Progress: 50% complete...")
                elif i == (3 * len(df)) // 4:
                    print(f"  Progress: 75% complete...")
            
            for j in range(i + 1, len(df)):
                row1 = df.iloc[i][cols_to_check]
                row2 = df.iloc[j][cols_to_check]
                
                avg_similarity, all_pass, field_details = self.calc_row_similarity_per_field(
                    row1, row2, threshold
                )
                
                #only consider it a duplicate if ALL fields pass threshold
                if all_pass and avg_similarity < 1.0:  #exclude exact matches
                    fuzzy_pairs.append((i, j, avg_similarity))
        
        print(f"  \u2713 Completed: {len(fuzzy_pairs)} fuzzy pairs found")
        
        self.fuzzy_duplicate_pairs = fuzzy_pairs
        
        #log detection
        self.duplicates_log.append({
            'operation': 'fuzzy_detection',
            'method': 'per_field_threshold',
            'columns_checked': subset if subset else 'all',
            'total_rows': len(df),
            'fuzzy_pairs_found': len(fuzzy_pairs),
            'threshold': threshold
        })
        
        return fuzzy_pairs
    
    
    def remove_fuzzy_duplicates(self, df: pd.DataFrame, fuzzy_pairs: Optional[List[Tuple[int, int, float]]] = None,
                               keep: str = 'first', interactive: bool = True) -> pd.DataFrame:
        """
        Remove fuzzy duplicate rows from dataframe with optional user review
        
        Args:
            df: Input dataframe
            fuzzy_pairs: List of (index1, index2, similarity) tuples (uses detected pairs if None)
            keep: Which duplicate to keep ('first' or 'last')
            interactive: If True, prompt user for review before removal
            
        Returns:
            DataFrame with fuzzy duplicates removed (or unchanged if user cancels)
        """
        if fuzzy_pairs is None:
            fuzzy_pairs = self.fuzzy_duplicate_pairs
        
        if not fuzzy_pairs:
            print("No fuzzy duplicates to remove")
            return df.copy()
        
        #interactive review prompt
        if interactive:
            print(f"\n{'='*70}")
            print("⚠️  FUZZY DUPLICATE REMOVAL WARNING")
            print(f"{'='*70}")
            print(f"Found {len(fuzzy_pairs)} fuzzy duplicate pairs.")
            print(f"Note: Fuzzy matching may produce false positives.")
            print(f"\nOptions:")
            print(f"  1. Review duplicates first")
            print(f"  2. Proceed with removal immediately")
            print(f"  3. Skip removal")
            
            while True:
                choice = input(f"\nEnter your choice (1/2/3): ").strip()
                
                if choice == '1':
                    #display all pairs for review
                    print(f"\n{'='*70}")
                    print("REVIEWING FUZZY DUPLICATE PAIRS")
                    print(f"{'='*70}\n")
                    
                    for i, (idx1, idx2, sim) in enumerate(sorted(fuzzy_pairs, key=lambda x: x[2], reverse=True), 1):
                        print(f"Pair {i}/{len(fuzzy_pairs)} (similarity: {sim:.2%}):")
                        print(f"  Row {idx1}: {df.iloc[idx1].to_dict()}")
                        print(f"  Row {idx2}: {df.iloc[idx2].to_dict()}")
                        print()
                        
                        if i % 10 == 0 and i < len(fuzzy_pairs):
                            cont = input(f"Continue viewing? ({i}/{len(fuzzy_pairs)} shown) [y/n]: ").strip().lower()
                            if cont != 'y':
                                break
                    
                    #prompt again after review
                    print(f"\nAfter review, do you want to proceed with removal?")
                    proceed = input("Enter 'yes' to proceed or 'no' to skip: ").strip().lower()
                    if proceed == 'yes':
                        break
                    else:
                        print("❌ Removal cancelled by user")
                        return df.copy()
                
                elif choice == '2':
                    print("✓ Proceeding with removal...")
                    break
                
                elif choice == '3':
                    print("❌ Removal cancelled by user")
                    return df.copy()
                
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
        
        #collect indices to drop
        indices_to_drop = set()
        
        for idx1, idx2, similarity in fuzzy_pairs:
            if keep == 'first':
                indices_to_drop.add(idx2)
            elif keep == 'last':
                indices_to_drop.add(idx1)
        
        #remove duplicates
        df_cleaned = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
        removed_count = len(indices_to_drop)
        self.duplicates_removed = removed_count
        self.fuzzy_duplicates_removed = removed_count
        
        print(f"✓ Removed {removed_count} fuzzy duplicate rows")
        
        #log removal
        self.duplicates_log.append({
            'operation': 'fuzzy_removal',
            'keep_strategy': keep,
            'original_rows': len(df),
            'duplicates_removed': self.duplicates_removed,
            'remaining_rows': len(df_cleaned),
            'reduction_percentage': round(self.duplicates_removed / len(df) * 100, 2) if len(df) > 0 else 0
        })
        
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None, 
                          keep: Literal['first', 'last', False] = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from dataframe
        
        Args:
            df: Input dataframe
            subset: List of columns to consider for duplicates (None = all columns)
            keep: Which duplicate to keep ('first', 'last', or False to drop all)
            
        Returns:
            DataFrame with duplicates removed
        """
        self.original_count = len(df)
        
        #remove duplicates
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        removed_count = self.original_count - len(df_cleaned)
        self.duplicates_removed = removed_count
        self.exact_duplicates_removed = removed_count
        
        #log removal
        self.duplicates_log.append({
            'operation': 'removal',
            'columns_checked': subset if subset else 'all',
            'keep_strategy': keep,
            'original_rows': self.original_count,
            'duplicates_removed': self.duplicates_removed,
            'remaining_rows': len(df_cleaned),
            'reduction_percentage': round(self.duplicates_removed / self.original_count * 100, 2) if self.original_count > 0 else 0
        })
        
        return df_cleaned
    
    def get_duplicate_groups(self, df: pd.DataFrame, subset: Optional[List[str]] = None):
        """
        Get all rows that have duplicates (including the first occurrence)
        Useful for inspection
        
        Args:
            df: Input dataframe
            subset: List of columns to consider for duplicates
            
        Returns:
            DataFrame with all duplicate groups
        """
        #find all rows that are duplicated (including first occurrence)
        duplicate_mask = df.duplicated(subset=subset, keep=False)
        duplicated_df = df[duplicate_mask]
        if subset is not None:
            return duplicated_df.sort_values(by=subset)  #type: ignore
        else:
            return duplicated_df.sort_values(by=df.columns.tolist())  #type: ignore
    
    def get_duplicates_summary(self) -> pd.DataFrame:
        """
        Return summary of all duplicate operations
        """
        if not self.duplicates_log:
            return pd.DataFrame(columns=['operation', 'columns_checked', 'duplicates_found'])
        
        return pd.DataFrame(self.duplicates_log)
    
    def save_duplicates_log(self, filepath: str):
        """
        Save duplicates log to JSON file
        
        Args:
            filepath: Path to save JSON log
        """
        # Extract summary from operations log
        exact_found = 0
        fuzzy_pairs_found = 0
        total_rows = 0
        
        for log_entry in self.duplicates_log:
            if log_entry.get('operation') == 'exact_detection':
                exact_found = log_entry.get('duplicates_found', 0)
                if total_rows == 0:
                    total_rows = log_entry.get('total_rows', 0)
            elif log_entry.get('operation') == 'fuzzy_detection':
                fuzzy_pairs_found = log_entry.get('fuzzy_pairs_found', 0)
                if total_rows == 0:
                    total_rows = log_entry.get('total_rows', 0)
        
        log_data = {
            'summary': {
                'total_rows_analyzed': total_rows,
                'exact_duplicates_found': exact_found,
                'exact_duplicates_removed': self.exact_duplicates_removed,
                'fuzzy_duplicate_pairs_found': fuzzy_pairs_found,
                'fuzzy_duplicates_removed': self.fuzzy_duplicates_removed,
                'total_duplicates_removed': self.exact_duplicates_removed + self.fuzzy_duplicates_removed
            },
            'operations': self.duplicates_log
        }
        
        #add fuzzy duplicate pairs if any were found
        if self.fuzzy_duplicate_pairs:
            log_data['fuzzy_duplicate_pairs'] = [
                {'index1': int(idx1), 'index2': int(idx2), 'similarity': float(sim)}
                for idx1, idx2, sim in self.fuzzy_duplicate_pairs[:10]  #save first 10
            ]
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"✓ Duplicates log saved to {filepath}")
    
    def print_summary(self, dataset_name: str = "Unknown"):
        """
        Print human-readable summary of duplicate detection/removal.
        
        Note: This method only prints the summary. To handle fuzzy duplicate
        removal with interactive review, call remove_fuzzy_duplicates() separately.
        
        Args:
            dataset_name: Name of the dataset being processed
        """
        print("=" * 70)
        print("DUPLICATE DETECTION SUMMARY".center(70))
        print("=" * 70)
        
        # Get stats from log
        exact_found = 0
        fuzzy_found = 0
        total_rows = 0
        exact_removed = 0
        
        for log_entry in self.duplicates_log:
            if log_entry.get('operation') == 'exact_detection':
                exact_found = log_entry.get('duplicates_found', 0)
                total_rows = log_entry.get('total_rows', 0)
            elif log_entry.get('operation') == 'fuzzy_detection':
                fuzzy_found = log_entry.get('fuzzy_pairs_found', 0)
                if total_rows == 0:
                    total_rows = log_entry.get('total_rows', 0)
            elif log_entry.get('operation') == 'removal':
                exact_removed = log_entry.get('duplicates_removed', 0)
        
        print(f"\n📊 DATASET INFO")
        print(f"   {'Dataset:':<30} {dataset_name:>15}")
        print(f"   {'Total Rows Tested:':<30} {total_rows:>15,}")
        
        print(f"\n🔍 EXACT DUPLICATE DETECTION")
        print(f"   {'Exact Duplicates Found:':<30} {exact_found:>15,}")
        if total_rows > 0:
            exact_pct = exact_found / total_rows * 100
            print(f"   {'Duplicate Percentage:':<30} {exact_pct:>14.2f}%")
        print(f"   {'Exact Duplicates Removed:':<30} {exact_removed:>15,}")
        
        print(f"\n🔎 FUZZY DUPLICATE DETECTION")
        print(f"   {'Fuzzy Threshold:':<30} {self.fuzzy_threshold:>15.2f}")
        print(f"   {'Fuzzy Duplicate Pairs Found:':<30} {len(self.fuzzy_duplicate_pairs):>15,}")
        
        # Show sample fuzzy pairs if found
        if self.fuzzy_duplicate_pairs:
            print(f"\n   📋 Sample Fuzzy Pairs (top 3):")
            for i, (idx1, idx2, sim) in enumerate(self.fuzzy_duplicate_pairs[:3], 1):
                print(f"      Pair {i}: Row {idx1} ↔ Row {idx2} (similarity: {sim:.2%})")
        
        print("\n" + "=" * 70)
