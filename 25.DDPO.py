import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

# Load data
data_path = r"C:\Users\wmqgxx\Desktop\上市企业数据集\5.特征选择后的数据集\train_t_3_selected.xlsx"
data = pd.read_excel(data_path, engine='openpyxl')

# Load feature weight data
weight_path = r"C:\Users\wmqgxx\Desktop\上市企业数据集\4_特征选择\回归系数表格\时间窗口_t3_回归系数.xlsx"
weight_data = pd.read_excel(weight_path, engine='openpyxl')

# Extract weights
weights = weight_data.iloc[:, 3].values
w = weight_data.iloc[:, 4].values

# Split features and labels
X = data.drop(columns=['Default Status'])
y = data['Default Status']

def weighted_euclidean_distances(X, weights):
    W = np.sqrt(weights)
    X_weighted = X * W
    n = X_weighted.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        diff = X_weighted - X_weighted.iloc[i]
        D[i] = np.sqrt(np.sum((diff ** 2), axis=1))
    return D

# Compute weighted distance matrix
D_w = weighted_euclidean_distances(X, weights)
distances = D_w[np.triu_indices(D_w.shape[0], k=1)]
distances_sorted = np.sort(distances)

# Define neighborhood radius
c = 0.5
threshold_index = int(c * len(distances_sorted))
r = distances_sorted[threshold_index]

# Find neighbors
neighbors = []
for i in range(len(X)):
    dist_to_all = D_w[i]
    neighbor_indices = np.where(dist_to_all <= r)[0]
    neighbors.append(neighbor_indices)

# Classify neighbors
S1j, S0j = [], []
for i, neighbor_indices in enumerate(neighbors):
    neighbor_indices = neighbor_indices[neighbor_indices != i]
    s1 = [i for i in neighbor_indices if y[i] == 1]
    s0 = [i for i in neighbor_indices if y[i] == 0]
    S1j.append(s1)
    S0j.append(s0)

# Label default instances
Snoisy, Ssafe, Sboundary = [], [], []
for i in range(len(S1j)):
    if y[i] == 1:
        if len(S1j[i]) == 0:
            Snoisy.append(i)
        elif len(S0j[i]) == 0:
            Ssafe.append(i)
        else:
            Sboundary.append(i)

# Compute boundary contributions
boundary_contributions = []
for i in Sboundary:
    d11 = np.min(D_w[i, S1j[i]]) if S1j[i] else float('inf')
    d10 = np.min(D_w[i, S0j[i]]) if S0j[i] else float('inf')
    len_s1 = len(S1j[i])
    len_s0 = len(S0j[i])
    Ci = (d11 / d10) + (len_s0 / (len_s1 + len_s0))
    boundary_contributions.append({
        'Company': i,
        'd11': d11,
        'd10': d10,
        'S1j': len_s1,
        'S0j': len_s0,
        'Contribution': Ci,
    })

# Compute sampling weights
total_contribution = sum([sample['Contribution'] for sample in boundary_contributions])
for sample in boundary_contributions:
    sample['SamplingWeight'] = sample['Contribution'] / total_contribution

# Determine number of synthetic samples
non_default_count = len(X[y == 0])
default_count = len(X[y == 1])
T = non_default_count - default_count
for sample in boundary_contributions:
    wi = sample['SamplingWeight']
    sample['RequiredSyntheticDefaults'] = int(wi * T)

# Generate new default samples
new_default_samples = []
for sample in boundary_contributions:
    if sample['RequiredSyntheticDefaults'] > 0:
        x1j_index = sample['Company']
        x1j = X.iloc[x1j_index].values
        for _ in range(sample['RequiredSyntheticDefaults']):
            neighbor_index = random.choice(S1j[x1j_index])
            x1l_prime = X.iloc[neighbor_index].values
            eg = random.random()
            new_default = x1j + eg * w * (x1l_prime - x1j)
            new_default_samples.append(new_default)

# Merge with original data
new_default_df = pd.DataFrame(new_default_samples, columns=X.columns).drop_duplicates()
new_default_df['(525)违约状态'] = 1
Sb_train = pd.concat([data, new_default_df], ignore_index=True)

# Save balanced dataset
output_path = r"C:\Users\wmqgxx\Desktop\train_t_3_balance0.5.xlsx"
Sb_train.to_excel(output_path, index=False)

# Save sampling details
sampling_summary = []
for sample in boundary_contributions:
    sampling_summary.append({
        'Company': sample['Company'],
        'SamplingWeight': sample['SamplingWeight'],
        'RequiredSyntheticDefaults': sample['RequiredSyntheticDefaults']
    })

print(f"Balanced training set saved to: {output_path}")
