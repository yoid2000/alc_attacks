import anonymeter_mods
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

class SrmAttack:
    def __init__(self,
                 df_original: pd.DataFrame,
                 df_control: pd.DataFrame,
                 df_synthetic: Union[pd.DataFrame, List[pd.DataFrame]],
                 results_path: str = None,
                 max_known_col_sets: int = 1000,
                 num_per_secret_attacks: int = 100,
                 num_rows_per_attack: int = 500,
                 ) -> None:
        # self.adf contains all of the dataframes needed for the attack, cleaned
        # up to work with ML modeling
        self.results_path = results_path
        self.max_known_col_sets = max_known_col_sets
        self.num_per_secret_attacks = num_per_secret_attacks
        self.num_rows_per_attack = num_rows_per_attack
        self.adf = DataFiles(df_original, df_control, df_synthetic)
        self.base_pred = BaselinePredictor(self.adf)
        # df_atk are the rows that will be the target of the attack
        self.df_atk = self.adf.orig.sample(len(self.adf.cntl))
        self.pred_res = PredictionResults(results_path = self.results_path)

    def run_auto_attack(self):
        '''
        Runs attacks against all categorical columns for a variety of known columns
        '''
        if self.results_path is None:
            # raise a value error
            raise ValueError("results_path must be set")
        # select a set of original rows to use for the attack
        df_atk = self.adf.orig.sample(len(self.adf.cntl))
        # Count the number of unique rows in df_atk
        known_column_sets = find_unique_column_sets(df_atk, max_sets = self.max_known_col_sets)
        print(f"Found {len(known_column_sets)} unique known column sets ")
        min_set_size = min([len(col_set) for col_set in known_column_sets])
        max_set_size = max([len(col_set) for col_set in known_column_sets])
        print(f"Minimum set size: {min_set_size}, Maximum set size: {max_set_size}")
        per_secret_column_sets = {}
        max_col_set_size = 0
        secret_cols = [col for col in self.adf.orig.columns if self.adf.is_categorical(col)]
        print(f"{len(secret_cols)} categorical columns of {len(self.adf.orig.columns)} columns: {secret_cols}")
        for secret_col in secret_cols:
            valid_known_column_sets = [col_set for col_set in known_column_sets if secret_col not in col_set]
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
                self.pred_res.summarize_results()
    
    def attack_known_cols_secret(self, secret_col: str,
                                 known_columns: List[str],
                                 df_base: pd.DataFrame,
                                 df_atk: pd.DataFrame) -> None:
        print(f"Attack secret column {secret_col}\n    assuming known columns {known_columns}")
        self.model_attack(secret_col, known_columns, df_base)
        self.simple_row_attack(secret_col, known_columns, df_atk)

    def model_attack(self,
                     secret_col: str,
                     known_columns: List[str],
                     df_base: pd.DataFrame) -> None:
        self.base_pred.build_model(known_columns, secret_col)
        # loop through the rows in df_base
        num_rows = 0
        for i, row in df_base.iterrows():
            if num_rows > self.num_rows_per_attack:
                break
            num_rows += 1
            # get the prediction for the row
            df_row = row[known_columns].to_frame().T
            predicted_value, proba = self.base_pred.predict(df_row)
            true_value = row[secret_col]
            decoded_predicted_value = self.adf.decode_value(secret_col, predicted_value)
            decoded_true_value = self.adf.decode_value(secret_col, true_value)
            self.pred_res.add_base_result(known_columns = known_columns,
                                     target_col = secret_col,
                                     predicted_value = decoded_predicted_value,
                                     true_value = decoded_true_value,
                                     prediction_proba = proba,
                                     fraction_agree = None
                                     )

    def simple_row_attack(self,
                          secret_col: str,
                          known_columns: List[str],
                          df_atk: pd.DataFrame) -> None:
        num_rows = 0
        for i, row in df_atk.iterrows():
            if num_rows > self.num_rows_per_attack:
                break
            num_rows += 1
            predictions = []
            for df_syn in self.adf.syn_list:
                # check if all known_columns and secret are in df_syn
                all_columns = known_columns + [secret_col]
                if not all(col in df_syn.columns for col in all_columns):
                    continue
                df_row = row[known_columns].to_frame().T
                ans_syn = anonymeter_mods.run_anonymeter_attack(
                                                targets=df_row,
                                                basis=df_syn,
                                                aux_cols=known_columns,
                                                secret=secret_col,
                                                regression=False)
                # Compute an answer based on the vanilla anonymeter attack
                pred_value_series = ans_syn['guess_series']
                pred_value = pred_value_series.iloc[0]
                predictions.append(pred_value.item())
            # determine the most frequent prediction in predictions, and the number
            # of times it appears
            counter = Counter(predictions)
            most_common_value, most_common_count = counter.most_common(1)[0]
            # determine the fraction of the predictions that are the most common
            fraction_agree = most_common_count / len(predictions)
            # get the true value of the secret column
            true_value = row[secret_col]
            decoded_predicted_value = self.adf.decode_value(secret_col, most_common_value)
            decoded_true_value = self.adf.decode_value(secret_col, true_value)
            self.pred_res.add_attack_result(known_columns = known_columns,
                                     target_col = secret_col,
                                     predicted_value = decoded_predicted_value,
                                     true_value = decoded_true_value,
                                     prediction_proba = None,
                                     fraction_agree = fraction_agree
                                     )


def find_valid_targets(df: pd.DataFrame, column: str) -> list:
    '''
    We don't want to attack values that are too rare, because we may have trouble
    getting a significant number of attacks. Values that are too common, on the other
    hand, are probably not sensitive and therefore not very interesting.
    '''
    value_counts = df[column].value_counts(normalize=True)
    valid_targets = value_counts[(value_counts > 0.002) & (value_counts < 0.60)].index.tolist()
    return valid_targets


def find_unique_column_sets(df: pd.DataFrame, max_sets: int = 1000) -> list:
    num_unique_rows = df.drop_duplicates().shape[0]
    column_sets = []
    stats = {}
    for col in df.columns:
        stats[col] = 0
    for r in range(1, len(df.columns) + 1):
        inactive = 0
        col_combs = list(combinations(df.columns, r))
        random.shuffle(col_combs)
        for cols in col_combs:
            inactive += 1
            if inactive > 50:
                # Don't continue working on overly small column sets
                break
            num_distinct = df[list(cols)].drop_duplicates().shape[0]
            if num_distinct == num_unique_rows:
                inactive = 0
                column_sets.append(cols)
                for col in cols:
                    stats[col] += 1
                if len(column_sets) >= max_sets:
                    pp.pprint(stats)
                    return column_sets
    return column_sets

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
        srm = SrmAttack(df_original=df_original,
                        df_control=df_control,
                        df_synthetic=syn_dfs,
                        results_path=results_path,
                        max_known_col_sets=100,
                        num_per_secret_attacks=2,
                        num_rows_per_attack=10,
                        )
    else:
        srm = SrmAttack(df_original=df_original,
                        df_control=df_control,
                        df_synthetic=syn_dfs,
                        results_path=results_path,
                        )
    srm.run_auto_attack()



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