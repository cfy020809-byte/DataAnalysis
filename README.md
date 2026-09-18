# DataAnalysis
import pandas as pd
df = pd.read_pickle(r'E:\datafx\LSWMD.pkl')
print('加载完成')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载全量
df_full = df[['waferMap', 'lotName', 'failureType']]

# 展平标签
def flatten_label(x):
    arr = np.asarray(x, dtype=object).ravel()
    return arr[0] if arr.size > 0 else 'Unlabeled'

df_full['failureType'] = df_full['failureType'].apply(flatten_label)

print('全量数据:', df_full.shape)
print(df_full['failureType'].value_counts())

# 分层抽样
labeled = df_full[df_full['failureType'] != 'Unlabeled'].sample(30000, random_state=42)
unlabeled = df_full[df_full['failureType'] == 'Unlabeled'].sample(50000, random_state=42)
sub = pd.concat([labeled, unlabeled]).sample(frac=1, random_state=42).reset_index(drop=True)

sub.to_pickle(r'E:\datafx\wm811k_sub.pkl')
print('子集已保存:', sub.shape)

del df_full, labeled, unlabeled, sub
