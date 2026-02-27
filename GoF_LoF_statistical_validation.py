import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random
from joblib import Parallel, delayed

file_path = "wagner_gain_loss_summary.csv"
df = pd.read_csv(file_path)

N_SHUFFLES = 10000
RANDOM_SEED = 42
N_THREADS = 30 

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def compute_auc_pvalue(events, total, n_shuffles=N_SHUFFLES):
    if events == 0 or events == total:
        return np.nan, np.nan

    y_true = [1] * events + [0] * (total - events)
    y_scores = [1] * events + [0] * (total - events)

    auc_obs = roc_auc_score(y_true, y_scores)

    auc_null = []
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(y_scores)
        auc_shuff = roc_auc_score(y_true, shuffled)
        auc_null.append(auc_shuff)

    p_value = np.mean([auc >= auc_obs for auc in auc_null])

    return auc_obs, p_value

def process_row(row):
    family = row['family']
    gains = row['gains']
    losses = row['losses']

    auc_gain, p_gain = (np.nan, np.nan)
    if gains >= 5 and gains < gains + losses:
        auc_gain, p_gain = compute_auc_pvalue(gains, gains + losses)

    auc_loss, p_loss = (np.nan, np.nan)
    if losses >= 5 and losses < gains + losses:
        auc_loss, p_loss = compute_auc_pvalue(losses, gains + losses)

    return {
        'family': family,
        'gains': gains,
        'losses': losses,
        'auc_gain': auc_gain,
        'p_value_gain': p_gain,
        'auc_loss': auc_loss,
        'p_value_loss': p_loss
    }

results = Parallel(n_jobs=N_THREADS)(
    delayed(process_row)(row) for _, row in tqdm(df.iterrows(), total=len(df))
)

df_results = pd.DataFrame(results)
df_results.to_csv("significant_gain_loss_OG.csv", index=False)

# Anteprima
print(df_results.head())

