"""
Duplicate Detector Module
Rule-based detection and removal of exact and fuzzy duplicate rows
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Literal
import json
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
    
    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Initialize duplicate detector
        
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0 to 1.0)
                           Default 0.85 means rows must be 85% similar to be considered duplicates
        """
        self.duplicates_log: List[Dict[str, Any]] = []
        self.duplicate_rows: Optional[pd.DataFrame] = None
        self.original_count: int = 0
        self.duplicates_found: int = 0
        self.duplicates_removed: int = 0
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_duplicate_pairs: List[Tuple[int, int, float]] = []  #(idx1, idx2, similarity)
    
    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None):
        """
        Detect duplicate rows in the dataframe
        
        Args:
            df: Input dataframe
            subset: List of columns to consider for duplicates (None = all columns)
            
        Returns:
            DataFrame containing only the duplicate rows
        """
        self.original_count = len(df)
        
        #find duplicates (marks all duplicates except first occurrence)
        duplicate_mask = df.duplicated(subset=subset, keep='first')
        self.duplicate_rows = df[duplicate_mask].copy()  #type: ignore
        self.duplicates_found = len(self.duplicate_rows) if self.duplicate_rows is not None else 0
        
        #log detection
        self.duplicates_log.append({
            'operation': 'detection',
            'columns_checked': subset if subset else 'all',
            'total_rows': self.original_count,
            'duplicates_found': self.duplicates_found,
            'duplicate_percentage': round(self.duplicates_found / self.original_count * 100, 2) if self.original_count > 0 else 0
        })
        
        return self.duplicate_rows
    
    def calc_row_similarity(self, row1: pd.Series, row2: pd.Series) -> float:
        """
        Calculate similarity between two rows
        
        Args:
            row1: First row
            row2: Second row
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if len(row1) != len(row2):
            return 0.0
        
        similarities = []
        
        for val1, val2 in zip(row1, row2):
            #handle missing values
            if pd.isna(val1) and pd.isna(val2):
                similarities.append(1.0)
                continue
            elif pd.isna(val1) or pd.isna(val2):
                similarities.append(0.0)
                continue
            
            #convert to strings for comparison
            str1 = str(val1).lower().strip()
            str2 = str(val2).lower().strip()
            
            #exact match
            if str1 == str2:
                similarities.append(1.0)
            else:
                #fuzzy match using Levenshtein distance
                similarity = Levenshtein.ratio(str1, str2)
                similarities.append(similarity)
        
        #return average similarity across all columns
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def calc_row_similarity_per_field(self, row1: pd.Series, row2: pd.Series, 
                                       per_field_threshold: float = 0.85) -> Tuple[float, bool, List[Tuple[str, float]]]:
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
                continue
            
            #convert to strings for comparison
            str1 = str(val1).lower().strip()
            str2 = str(val2).lower().strip()
            
            #calculate similarity
            if str1 == str2:
                similarity = 1.0
            else:
                #fuzzy match using Levenshtein distance
                similarity = Levenshtein.ratio(str1, str2)
            
            field_similarities.append((col_name, similarity))
            
            #check if this field passes the threshold
            if similarity < per_field_threshold:
                all_pass = False
        
        #calculate average similarity
        avg_similarity = sum(sim for _, sim in field_similarities) / len(field_similarities) if field_similarities else 0.0
        
        return avg_similarity, all_pass, field_similarities
    
    def detect_fuzzy_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None,
                               threshold: Optional[float] = None,
                               use_per_field_threshold: bool = True) -> List[Tuple[int, int, float]]:
        """
        Detect fuzzy duplicate rows in the dataframe
        
        Args:
            df: Input dataframe
            subset: List of columns to consider for duplicates (None = all columns)
            threshold: Similarity threshold (overrides instance threshold if provided)
            use_per_field_threshold: If True, ALL fields must individually meet threshold
                                    If False, uses average similarity (old behavior)
            
        Returns:
            List of tuples (index1, index2, similarity_score) for detected fuzzy duplicates
        """
        self.original_count = len(df)
        threshold = threshold if threshold is not None else self.fuzzy_threshold
        
        fuzzy_pairs = []
        cols_to_check = subset if subset else df.columns.tolist()
        
        #compare each pair of rows
        print(f"Checking {len(df)} rows for fuzzy duplicates (threshold={threshold})...")
        if use_per_field_threshold:
            print(f"  Using per-field threshold: ALL fields must be ≥{threshold}")
        else:
            print(f"  Using average similarity threshold")
        
        for i in range(len(df)):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(df)} rows...")
            
            for j in range(i + 1, len(df)):
                row1 = df.iloc[i][cols_to_check]
                row2 = df.iloc[j][cols_to_check]
                
                if use_per_field_threshold:
                    avg_similarity, all_pass, field_details = self.calc_row_similarity_per_field(
                        row1, row2, threshold
                    )
                    
                    #only consider it a duplicate if ALL fields pass threshold
                    if all_pass and avg_similarity < 1.0:  #exclude exact matches
                        fuzzy_pairs.append((i, j, avg_similarity))
                else:
                    #old behavior: use average similarity
                    similarity = self.calc_row_similarity(row1, row2)
                    
                    if similarity >= threshold and similarity < 1.0:
                        fuzzy_pairs.append((i, j, similarity))
        
        self.fuzzy_duplicate_pairs = fuzzy_pairs
        self.duplicates_found = len(fuzzy_pairs)
        
        #log detection
        self.duplicates_log.append({
            'operation': 'fuzzy_detection',
            'method': 'per_field_threshold' if use_per_field_threshold else 'average_similarity',
            'columns_checked': subset if subset else 'all',
            'total_rows': self.original_count,
            'fuzzy_pairs_found': len(fuzzy_pairs),
            'threshold': threshold
        })
        
        print(f"✓ Found {len(fuzzy_pairs)} fuzzy duplicate pairs")
        
        return fuzzy_pairs
    
    def remove_fuzzy_duplicates(self, df: pd.DataFrame, fuzzy_pairs: Optional[List[Tuple[int, int, float]]] = None,
                               keep: str = 'first') -> pd.DataFrame:
        """
        Remove fuzzy duplicate rows from dataframe
        
        Args:
            df: Input dataframe
            fuzzy_pairs: List of (index1, index2, similarity) tuples (uses detected pairs if None)
            keep: Which duplicate to keep ('first' or 'last')
            
        Returns:
            DataFrame with fuzzy duplicates removed
        """
        if fuzzy_pairs is None:
            fuzzy_pairs = self.fuzzy_duplicate_pairs
        
        if not fuzzy_pairs:
            print("No fuzzy duplicates to remove")
            return df.copy()
        
        #collect indices to drop
        indices_to_drop = set()
        
        for idx1, idx2, similarity in fuzzy_pairs:
            if keep == 'first':
                indices_to_drop.add(idx2)
            elif keep == 'last':
                indices_to_drop.add(idx1)
        
        #remove duplicates
        df_cleaned = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
        self.duplicates_removed = len(indices_to_drop)
        
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
        
        #detect duplicates first
        duplicate_mask = df.duplicated(subset=subset, keep=keep)
        self.duplicates_found = duplicate_mask.sum()
        
        #store duplicate rows before removal
        self.duplicate_rows = df[duplicate_mask].copy()  #type: ignore
        
        #remove duplicates
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        self.duplicates_removed = self.original_count - len(df_cleaned)
        
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
        log_data = {
            'original_row_count': self.original_count,
            'duplicates_found': self.duplicates_found,
            'duplicates_removed': self.duplicates_removed,
            'operations': self.duplicates_log
        }
        
        #add sample of duplicate rows if any were found
        if self.duplicate_rows is not None and len(self.duplicate_rows) > 0:
            sample_df = self.duplicate_rows.head(10)  #type: ignore
            log_data['sample_duplicates'] = list(sample_df.T.to_dict().values())
        
        #add fuzzy duplicate pairs if any were found
        if self.fuzzy_duplicate_pairs:
            log_data['fuzzy_duplicate_pairs'] = [
                {'index1': int(idx1), 'index2': int(idx2), 'similarity': float(sim)}
                for idx1, idx2, sim in self.fuzzy_duplicate_pairs[:10]  #save first 10
            ]
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"✓ Duplicates log saved to {filepath}")
    
    def print_summary(self):
        """
        Print human-readable summary of duplicate detection/removal
        """
        print("=" * 70)
        print("DUPLICATE DETECTION SUMMARY".center(70))
        print("=" * 70)
        
        print(f"\n📊 OVERVIEW")
        print(f"   {'Original Rows:':<25} {self.original_count:>15,}")
        print(f"   {'Exact Duplicates Found:':<25} {self.duplicates_found:>15,}")
        print(f"   {'Fuzzy Duplicate Pairs:':<25} {len(self.fuzzy_duplicate_pairs):>15,}")
        print(f"   {'Duplicates Removed:':<25} {self.duplicates_removed:>15,}")
        
        if self.original_count > 0:
            dup_pct = self.duplicates_found / self.original_count * 100
            print(f"   {'Exact Dup Percentage:':<25} {dup_pct:>14.2f}%")
        
        #show operations log
        if self.duplicates_log:
            print(f"\n🔧 OPERATIONS LOG")
            print(f"   {'Operation':<20} {'Columns':<15} {'Result':<30}")
            print(f"   {'-'*20} {'-'*15} {'-'*30}")
            
            for log in self.duplicates_log:
                op = log['operation']
                
                if op == 'fuzzy_detection':
                    cols = 'similarity'
                    result = f"{log['fuzzy_pairs_found']:,} similar pairs (threshold={log.get('threshold', 0.85)})"
                elif op == 'fuzzy_removal':
                    cols = log.get('keep_strategy', 'first')
                    result = f"Removed {log['duplicates_removed']:,} fuzzy duplicates"
                else:
                    cols = 'all' if log['columns_checked'] == 'all' else str(len(log['columns_checked'])) + ' cols'
                    if op == 'detection':
                        result = f"{log['duplicates_found']:,} duplicates ({log['duplicate_percentage']}%)"
                    else:
                        result = f"Removed {log['duplicates_removed']:,} rows"
                
                print(f"   {op:<20} {cols:<15} {result:<30}")
        
        #show sample duplicates if any
        if self.duplicate_rows is not None and len(self.duplicate_rows) > 0:
            print(f"\n📋 SAMPLE EXACT DUPLICATE ROWS (first 5)")
            print(self.duplicate_rows.head().to_string())
        
        #show sample fuzzy duplicates if any
        if self.fuzzy_duplicate_pairs:
            print(f"\n🔍 SAMPLE FUZZY DUPLICATE PAIRS (first 5)")
            for idx1, idx2, sim in self.fuzzy_duplicate_pairs[:5]:
                print(f"   Row {idx1} ↔ Row {idx2}: {sim:.2%} similar")
        
        print("\n" + "=" * 70)
