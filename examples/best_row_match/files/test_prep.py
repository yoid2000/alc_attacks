import os
import pandas as pd
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams

# read tx2019.csv
df_initial = pd.read_csv('tx2019.csv')
df_control = df_initial.sample(n=1000, random_state=42)
df_original = df_initial.drop(df_control.index)

input_path = os.path.join('attack_files', 'inputs')
# create directory if it does not exist
os.makedirs(input_path, exist_ok=True)
# create original.csv and control.csv
df_original.to_csv(os.path.join(input_path, 'original.csv'), index=False)
df_control.to_csv(os.path.join(input_path, 'control.csv'), index=False)

synthetic_path = os.path.join(input_path, 'synthetic_files')
os.makedirs(synthetic_path, exist_ok=True)

for i in range(5):
    syn = Synthesizer(df_original)
    syn = Synthesizer(df_original,
                      anonymization_params=AnonymizationParams(salt=bytes(str(i), 'utf-8')))
    df_syn = syn.sample()
    syn_path = os.path.join(synthetic_path, f'syn_{i}.csv')
    df_syn.to_csv(syn_path, index=False)
