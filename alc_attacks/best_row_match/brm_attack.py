from matching_routines import find_best_matches, modal_fraction, best_match_confidence
import argparse
import os
import pandas as pd
import sys
import random
from typing import List, Union
from itertools import combinations
from collections import Counter
from anonymity_loss_coefficient.anonymity_loss_coefficient import AnonymityLossCoefficient, DataFiles, BaselinePredictor, PredictionResults
import pprint

pp = pprint.PrettyPrinter(indent=4)

test_params = False

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
        # self.adf contains all of the dataframes needed for the attack, cleaned
        # up to work with ML modeling
        self.results_path = results_path
        self.max_known_col_sets = max_known_col_sets
        self.num_per_secret_attacks = num_per_secret_attacks
        self.max_rows_per_attack = max_rows_per_attack
        self.min_rows_per_attack = min_rows_per_attack
        self.min_positive_predictions = min_positive_predictions
        self.confidence_interval_tolerance = confidence_interval_tolerance
        self.confidence_level = confidence_level
        self.original_columns = df_original.columns.tolist()
        print(f"Original columns: {self.original_columns}")
        self.adf = DataFiles(df_original, df_control, df_synthetic)
        self.base_pred = BaselinePredictor(self.adf)
        # df_atk are the rows that will be the target of the attack
        self.df_atk = self.adf.orig.sample(len(self.adf.cntl))
        self.pred_res = PredictionResults(results_path = self.results_path,
                                          attack_name = attack_name)
        # The known columns are the pre-discretized continuous columns and categorical
        # columns (i.e. all original columns). The secret columns are the discretized
        # continuous columns and categorical columns.
        self.known_columns = self.original_columns
        self.secret_cols = [self.adf.get_discretized_column(col) for col in self.original_columns]
        print(f"There are {len(self.known_columns)} potential known columns:")
        print(self.known_columns)
        print(f"There are {len(self.secret_cols)} potential secret columns:")
        print(self.secret_cols)
        print("Columns are classified as:")
        pp.pprint(self.adf.column_classification)

    def run_all_columns_attack(self):
        '''
        Runs attacks assuming all columns except secret are known
        '''
        # select a set of original rows to use for the attack
        df_atk = self.adf.orig.sample(len(self.adf.cntl))
        for secret_col in self.secret_cols:
            known_columns = [col for col in self.known_columns if col != self.adf.get_pre_discretized_column(secret_col)]
            self.attack_known_cols_secret(secret_col, known_columns, self.adf.cntl, df_atk)
            self.pred_res.summarize_results(with_plot=True)

    def run_auto_attack(self):
        '''
        Runs attacks against all categorical columns for a variety of known columns
        '''
        if self.results_path is None:
            # raise a value error
            raise ValueError("results_path must be set")
        self.run_all_columns_attack()
        # select a set of original rows to use for the attack
        df_atk = self.adf.orig.sample(len(self.adf.cntl))
        # Count the number of unique rows in df_atk
        known_column_sets = self.find_unique_column_sets(self.adf.orig, max_sets = self.max_known_col_sets)
        print(f"Found {len(known_column_sets)} unique known column sets ")
        min_set_size = min([len(col_set) for col_set in known_column_sets])
        max_set_size = max([len(col_set) for col_set in known_column_sets])
        print(f"Minimum set size: {min_set_size}, Maximum set size: {max_set_size}")
        per_secret_column_sets = {}
        max_col_set_size = 0
        for secret_col in self.secret_cols:
            valid_known_column_sets = [col_set for col_set in known_column_sets if self.adf.get_pre_discretized_column(secret_col) not in col_set]
            sampled_known_column_sets = random.sample(valid_known_column_sets,
                                              min(self.num_per_secret_attacks, len(valid_known_column_sets)))
            max_col_set_size = max(max_col_set_size, len(sampled_known_column_sets))
            valid_targets = find_valid_targets(self.adf.orig, secret_col)
            df_base = self.adf.cntl[self.adf.cntl[secret_col].isin(valid_targets)]
            df_atk = self.df_atk[self.df_atk[secret_col].isin(valid_targets)]
            per_secret_column_sets[secret_col] = {'known_column_sets': sampled_known_column_sets, 'df_base': df_base, 'df_atk': df_atk}
        for i in range(max_col_set_size):
            for secret_col, info in per_secret_column_sets.items():
                if i < len(info['known_column_sets']):
                    self.attack_known_cols_secret(secret_col,
                                                  list(info['known_column_sets'][i]),
                                                  info['df_base'],
                                                  info['df_atk'])
                # After each round of secret columns, we save the data and produce
                # a report
                self.pred_res.summarize_results(with_plot=True)
    
    def attack_known_cols_secret(self, secret_col: str,
                                 known_columns: List[str],
                                 df_base_in: pd.DataFrame,
                                 df_atk_in: pd.DataFrame) -> None:
        print(f"Attack secret column {secret_col}\n    assuming {len(known_columns)} known columns {known_columns}")
        if self.adf.get_pre_discretized_column(secret_col) in known_columns:
            raise ValueError(f"Secret column {secret_col} is in known columns")
        for known_column in known_columns:
            if self.adf.get_discretized_column(known_column) == secret_col:
                raise ValueError(f"Secret column {secret_col} is in known columns")
        # Shuffle df_base and df_atk to avoid any bias
        df_base = df_base_in.sample(frac=1).reset_index(drop=True)
        df_atk = df_atk_in.sample(frac=1).reset_index(drop=True)
        self.base_pred.build_model(known_columns, secret_col)
        for i in range(min(len(df_base), len(df_atk), self.max_rows_per_attack)):
            # Get one base and attack measure at a time, and continue until we have
            # enough confidence in the results
            base_row = df_base.iloc[i]
            self.model_attack(base_row, secret_col, known_columns)
            atk_row = df_atk.iloc[i]
            self.best_row_attack(atk_row, secret_col, known_columns)
            if i >= 50 and i % 10 == 0:
                # Check for confidence after every 10 attack predictions
                ci_info = self.pred_res.get_ci()
                cii = ci_info['base']
                pos_pred_count = round(cii['n'] * cii['prec'])
                if cii['ci_high'] - cii['ci_low'] <= self.confidence_interval_tolerance and pos_pred_count >= self.min_positive_predictions:
                    print(f"Base confidence interval ({round(cii['ci_low'],2)}, {round(cii['ci_high'],2)}) is within tolerance after {i+1} attacks on precision {round(cii['prec'],2)}")
                    break
                cii = ci_info['attack']
                pos_pred_count = round(cii['n'] * cii['prec'])
                if cii['ci_high'] - cii['ci_low'] <= self.confidence_interval_tolerance and pos_pred_count >= self.min_positive_predictions:
                    print(f"Attack confidence interval ({round(cii['ci_low'],2)}, {round(cii['ci_high'],2)}) is within tolerance after {i+1} attacks on precision {round(cii['prec'],2)}")
                    break

    def model_attack(self, row: pd.Series,
                     secret_col: str,
                     known_columns: List[str]) -> None:
        # get the prediction for the row
        df_row = row[known_columns].to_frame().T
        predicted_value, proba = self.base_pred.predict(df_row)
        true_value = row[secret_col]
        decoded_predicted_value = self.adf.decode_value(secret_col, predicted_value)
        decoded_true_value = self.adf.decode_value(secret_col, true_value)
        self.pred_res.add_base_result(known_columns = known_columns,
                                    secret_col = secret_col,
                                    predicted_value = decoded_predicted_value,
                                    true_value = decoded_true_value,
                                    base_confidence = proba,
                                    )

    def best_row_attack(self, row: pd.Series,
                          secret_col: str,
                          known_columns: List[str]) -> None:
        best_confidence = -1
        best_pred_value = None
        for df_syn in self.adf.syn_list:
            # Check if secret_col is in df_syn
            if secret_col not in df_syn.columns:
                continue
            # Make sure there is at least one known column in df_syn
            shared_known_columns = list(set(known_columns) & set(df_syn.columns))
            if len(shared_known_columns) == 0:
                continue
            df_query = row[shared_known_columns].to_frame().T
            idx, min_gower_distance = find_best_matches(df_query=df_query,
                                                        df_candidates=df_syn,
                                                        column_classifications=self.adf.column_classification,
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
        true_value = row[secret_col]
        decoded_true_value = self.adf.decode_value(secret_col, true_value)
        decoded_predicted_value = self.adf.decode_value(secret_col, best_pred_value)
        self.pred_res.add_attack_result(known_columns = known_columns,
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


def find_valid_targets(df: pd.DataFrame, column: str) -> list:
    '''
    We don't want to attack values that are too rare, because we may have trouble
    getting a significant number of attacks. Values that are too common, on the other
    hand, are probably not sensitive and therefore not very interesting.
    '''
    value_counts = df[column].value_counts(normalize=True)
    valid_targets = value_counts[(value_counts > 0.002) & (value_counts < 0.60)].index.tolist()
    return valid_targets

def print_col_info(df):
    for col in df.columns:
        print(df[col].head())
        print('Num unique:', df[col].nunique())
        print(df[col].describe())

def establish_bins(df, types):
    bins_dict = {}
    for col, col_type in types.items():
        if col_type == 'continuous' and df[col].nunique() > 50:
            # Establish the bins for the column
            bins = pd.cut(df[col], bins=20, retbins=True)[1]
            bins_dict[col] = bins
    return bins_dict

def apply_bins(df, bins_dict):
    for col, bins in bins_dict.items():
        if col in df.columns:
            # Apply the established bins to the column
            df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    return df

def run_attacks(attack_files_path):
    #split attack_files_path into the path and the attack_files directory
    attack_dir_name = os.path.split(attack_files_path)[-1]
    # Check that there is indeed a directory at attack_files_path
    if not os.path.isdir(attack_files_path):
        print(f"Error: {attack_files_path} is not a directory")
        sys.exit(1)
    inputs_path = os.path.join(attack_files_path, 'inputs')
    if not os.path.exists(inputs_path) or not os.path.isdir(inputs_path):
        print(f"Error: {inputs_path} does not exist or is not a directory")
        print(f"Your test files should be in {inputs_path}")
        sys.exit(1)
    control_data_path = os.path.join(inputs_path, 'control.csv')
    # read the control data
    try:
        df_control = pd.read_csv(control_data_path)
    except Exception as e:
        print(f"Error reading {control_data_path}")
        print(f"Error: {e}")
        sys.exit(1)
    if not os.path.exists(control_data_path):
        print(f"Error: {control_data_path} does not exist")
        sys.exit(1)
    original_data_path = os.path.join(inputs_path, 'original.csv')
    try:
        df_original = pd.read_csv(original_data_path)
    except Exception as e:
        print(f"Error reading {original_data_path}")
        print(f"Error: {e}")
        sys.exit(1)
    if not os.path.exists(original_data_path):
        print(f"Error: {original_data_path} does not exist")
        sys.exit(1)
    synthetic_path = os.path.join(inputs_path, 'synthetic_files')
    if not os.path.exists(synthetic_path) or not os.path.isdir(synthetic_path):
        print(f"Error: {synthetic_path} does not exist or is not a directory")
        sys.exit(1)
    syn_dfs = []
    for file in os.listdir(synthetic_path):
        if file.endswith('.csv'):
            syn_dfs.append(pd.read_csv(os.path.join(synthetic_path, file)))
    results_path = os.path.join(attack_files_path, 'results')
    if test_params:
        brm = BrmAttack(df_original=df_original,
                        df_control=df_control,
                        df_synthetic=syn_dfs,
                        results_path=results_path,
                        max_known_col_sets=100,
                        num_per_secret_attacks=2,
                        max_rows_per_attack=10,
                        min_rows_per_attack=5,
                        attack_name = attack_dir_name,
                        )
    else:
        brm = BrmAttack(df_original=df_original,
                        df_control=df_control,
                        df_synthetic=syn_dfs,
                        results_path=results_path,
                        attack_name = attack_dir_name,
                        )
    brm.run_auto_attack()




def main():
    parser = argparse.ArgumentParser(description="Run different anonymeter_plus commands.")
    parser.add_argument("command", help="'attack' to run make_config(), 'stats' to run attack_stats(), or 'plots' to run do_plots()")
    parser.add_argument("attack_files_path", nargs='?', help="Optional path for attack files, used with the 'attack' command")

    args = parser.parse_args()

    if args.command == 'attack':
        attack_files_path = args.attack_files_path
        if attack_files_path:
            print(f"Using attack files path: {attack_files_path}")
        else:
            attack_files_path = 'attack_files'
        run_attacks(attack_files_path)
    elif args.command == 'stats':
        attack_stats()
    elif args.command == 'plots':
        do_plots()
    else:
        raise ValueError(f"Unrecognized command: {args.command}")

    # Print the attack_files_path to verify
    print(f"attack_files_path: {attack_files_path}")

if __name__ == "__main__":
    main()