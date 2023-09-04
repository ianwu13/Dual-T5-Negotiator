print('SCRIPT SHOULD NOT BE RE-RUN')
exit()

import pandas as pd
import csv


DATA_FILE = 'baseline_casino_opponent_pref.csv'
SAVE_DIR = './splits/casino_opponent_pref'

EVAL_SAMPLES = 1000
TEST_SAMPLES = 1000

df = pd.read_csv(DATA_FILE).sample(frac=1, random_state=99) # sample for shuffling

df_eval = df.iloc[:EVAL_SAMPLES, :]
df.drop(index=df.index[:EVAL_SAMPLES], axis=0, inplace=True)

df_test = df.iloc[:TEST_SAMPLES, :]
df.drop(index=df.index[:TEST_SAMPLES], axis=0, inplace=True)

df_eval.to_csv(f'{SAVE_DIR}/eval.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
df_test.to_csv(f'{SAVE_DIR}/test.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
df.to_csv(f'{SAVE_DIR}/train.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
