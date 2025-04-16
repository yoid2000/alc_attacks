from anonymity_loss_coefficient import ALCManager
from alc_attacks.best_row_match.matching_routines import find_best_matches, modal_fraction, best_match_confidence
import pandas as pd
import random
from typing import List, Union
from itertools import combinations
import pprint

pp = pprint.PrettyPrinter(indent=4)

class BrmAttack:
    def __init__(self,
                 df_original: pd.DataFrame,
                 df_control: pd.DataFrame,
                 df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                 results_path: str = None,
                 max_known_col_sets: int = 1000,
                 num_per_secret_attacks: int = 100,
                 max_rows_per_attack: int = 500,
                 min_rows_per_attack: int = 50,
                 min_positive_predictions: int = 5,
                 confidence_interval_tolerance: float = 0.1,
                 confidence_level: float = 0.95,
                 attack_name: str = '',
                 ) -> None:
        # up to work with ML modeling
        self.results_path = results_path
        self.max_known_col_sets = max_known_col_sets
        self.num_per_secret_attacks = num_per_secret_attacks
        self.max_rows_per_attack = max_rows_per_attack
        self.min_rows_per_attack = min_rows_per_attack
        self.min_positive_predictions = min_positive_predictions
        self.confidence_interval_tolerance = confidence_interval_tolerance
        self.confidence_level = confidence_level
        self.results_path = results_path
        self.attack_name = attack_name
        self.original_columns = df_original.columns.tolist()
        print(f"Original columns: {self.original_columns}")
        self.alcm = ALCManager(df_original, df_control, df_synthetic)
        # The known columns are the pre-discretized continuous columns and categorical
        # columns (i.e. all original columns). The secret columns are the discretized
        # continuous columns and categorical columns.
        self.known_columns = self.original_columns
        self.secret_cols = [self.alcm.get_discretized_column(col) for col in self.original_columns]
        print(f"There are {len(self.known_columns)} potential known columns:")
        print(self.known_columns)
        print(f"There are {len(self.secret_cols)} potential secret columns:")
        print(self.secret_cols)
        print("Columns are classified as:")
        pp.pprint(self.alcm.get_column_classification_dict())

    def run_all_columns_attack(self):
        '''
        Runs attacks assuming all columns except secret are known
        '''
        # select a set of original rows to use for the attack
        for secret_col in self.secret_cols:
            known_columns = [col for col in self.known_columns if col != self.alcm.get_pre_discretized_column(secret_col)]
            self.attack_known_cols_secret(secret_col, known_columns, self.alcm.df.cntl)
            self.alcm.summarize_results(results_path = self.results_path,
                                          attack_name = self.attack_name, with_plot=True)

    def run_auto_attack(self):
        '''
        Runs attacks against all categorical columns for a variety of known columns
        '''
        if self.results_path is None:
            # raise a value error
            raise ValueError("results_path must be set")
        self.run_all_columns_attack()
        known_column_sets = self.find_unique_column_sets(self.alcm.df.orig, max_sets = self.max_known_col_sets)
        print(f"Found {len(known_column_sets)} unique known column sets ")
        min_set_size = min([len(col_set) for col_set in known_column_sets])
        max_set_size = max([len(col_set) for col_set in known_column_sets])
        print(f"Minimum set size: {min_set_size}, Maximum set size: {max_set_size}")
        per_secret_column_sets = {}
        max_col_set_size = 0
        for secret_col in self.secret_cols:
            valid_known_column_sets = [col_set for col_set in known_column_sets if self.alcm.get_pre_discretized_column(secret_col) not in col_set]
            sampled_known_column_sets = random.sample(valid_known_column_sets,
                                              min(self.num_per_secret_attacks, len(valid_known_column_sets)))
            max_col_set_size = max(max_col_set_size, len(sampled_known_column_sets))
            valid_secret_targets = find_valid_secret_targets(self.alcm.df.orig, secret_col)
            df_cntl = self.alcm.df.cntl[self.alcm.df.cntl[secret_col].isin(valid_secret_targets)]
            per_secret_column_sets[secret_col] = {'known_column_sets': sampled_known_column_sets, 'df_cntl': df_cntl}
        for i in range(max_col_set_size):
            for secret_col, info in per_secret_column_sets.items():
                if i < len(info['known_column_sets']):
                    self.attack_known_cols_secret(secret_col,
                                                  list(info['known_column_sets'][i]),
                                                  info['df_cntl'])
                # After each round of secret columns, we save the data and produce
                # a report
                self.alcm.summarize_results(results_path = self.results_path,
                                          attack_name = self.attack_name, with_plot=True)
    
    def attack_known_cols_secret(self, secret_col: str,
                                 known_columns: List[str],
                                 df_cntl_in: pd.DataFrame) -> None:
        print(f"Attack secret column {secret_col}\n    assuming {len(known_columns)} known columns {known_columns}")
        if self.alcm.get_pre_discretized_column(secret_col) in known_columns:
            raise ValueError(f"Secret column {secret_col} is in known columns")
        for known_column in known_columns:
            if self.alcm.get_discretized_column(known_column) == secret_col:
                raise ValueError(f"Secret column {secret_col} is in known columns")
        # Shuffle to avoid any bias
        df_cntl = df_cntl_in.sample(frac=1).reset_index(drop=True)
        self.alcm.build_model(known_columns, secret_col)
        for i in range(min(len(df_cntl), self.max_rows_per_attack)):
            # Get one base and attack measure at a time, and continue until we have
            # enough confidence in the results
            atk_row = df_cntl.iloc[[i]]
            self.model_attack(atk_row, secret_col, known_columns)
            self.best_row_attack(atk_row, secret_col, known_columns)
            halt_ok, info, reason = self.alcm.ok_to_halt()
            if halt_ok:
                print(f'Ok to halt after {i} attacks with ALC {info['alc']:.2f} and reason: "{reason}"\n')
                return
        print(f"Halt conditions never reached after {len(df_cntl)} attacks.")
        pp.pprint(info)


    def model_attack(self, row: pd.DataFrame,
                     secret_col: str,
                     known_columns: List[str]) -> None:
        # get the prediction for the row
        df_row = row[known_columns]  # This is already a DataFrame
        predicted_value, proba = self.alcm.predict(df_row)
        true_value = row[secret_col].iloc[0]
        decoded_predicted_value = self.alcm.decode_value(secret_col, predicted_value)
        decoded_true_value = self.alcm.decode_value(secret_col, true_value)
        self.alcm.add_base_result(known_columns = known_columns,
                                    secret_col = secret_col,
                                    predicted_value = decoded_predicted_value,
                                    true_value = decoded_true_value,
                                    base_confidence = proba,
                                    )

    def best_row_attack(self, row: pd.DataFrame,
                          secret_col: str,
                          known_columns: List[str]) -> None:
        best_confidence = -1
        best_pred_value = None
        for df_syn in self.alcm.df.syn_list:
            # Check if secret_col is in df_syn
            if secret_col not in df_syn.columns:
                continue
            # Make sure there is at least one known column in df_syn
            shared_known_columns = list(set(known_columns) & set(df_syn.columns))
            if len(shared_known_columns) == 0:
                continue
            df_query = row[shared_known_columns]
            idx, min_gower_distance = find_best_matches(df_query=df_query,
                                                        df_candidates=df_syn,
                                                        column_classifications=self.alcm.get_column_classification_dict(),
                                                        columns=shared_known_columns)
            number_of_min_gower_distance_matches = len(idx)
            this_pred_value, modal_count = modal_fraction(df_candidates=df_syn,
                                                     idx=idx, column=secret_col)
            this_modal_fraction = modal_count / number_of_min_gower_distance_matches
            this_confidence = best_match_confidence(
                                          gower_distance=min_gower_distance,
                                          modal_fraction=this_modal_fraction,
                                          match_count=number_of_min_gower_distance_matches)
            if this_confidence > best_confidence:
                best_confidence = this_confidence
                best_pred_value = this_pred_value
        true_value = row[secret_col].iloc[0]
        decoded_true_value = self.alcm.decode_value(secret_col, true_value)
        decoded_predicted_value = self.alcm.decode_value(secret_col, best_pred_value)
        self.alcm.add_attack_result(known_columns = known_columns,
                                    secret_col = secret_col,
                                    true_value = decoded_true_value,
                                    predicted_value = decoded_predicted_value,
                                    attack_confidence = best_confidence
                                    )


    def find_unique_column_sets(self, df: pd.DataFrame, max_sets: int = 1000) -> list:
        num_unique_rows = df.drop_duplicates().shape[0]
        column_sets = []
        stats = {}
        for col in self.known_columns:
            stats[col] = 0
        for r in range(1, len(self.known_columns) + 1):
            inactive = 0
            col_combs = list(combinations(self.known_columns, r))
            random.shuffle(col_combs)
            for cols in col_combs:
                inactive += 1
                if inactive > 50:
                    # Don't continue working on overly small column sets
                    break
                num_distinct = df[list(cols)].drop_duplicates().shape[0]
                # We want a given known columns set of values to be unique with high probability
                if num_distinct >= (num_unique_rows * 0.95):
                    inactive = 0
                    column_sets.append(cols)
                    for col in cols:
                        stats[col] += 1
                    if len(column_sets) >= max_sets:
                        pp.pprint(stats)
                        return column_sets
        return column_sets


def find_valid_secret_targets(df: pd.DataFrame, column: str) -> list:
    '''
    We don't want to attack values that are too rare, because we may have trouble
    getting a significant number of attacks. Values that are too common, on the other
    hand, are probably not sensitive and therefore not very interesting.
    '''
    value_counts = df[column].value_counts(normalize=True)
    valid_secret_targets = value_counts[(value_counts > 0.002) & (value_counts < 0.60)].index.tolist()
    return valid_secret_targets