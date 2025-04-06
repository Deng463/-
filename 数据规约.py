import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# 数据加载与预处理
file_path = 'C:/Users/86185/PycharmProjects/数据规约/loan-data.csv'
data = pd.read_csv(file_path)

# 数据预处理
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# 填充缺失值
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# 独热编码（排除目标变量）
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(data[categorical_cols[:-1]])  # 假设最后一列是目标变量
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols[:-1]))

# 3. 特征选择
X = pd.concat([data[numeric_cols], encoded_categorical_df], axis=1)
y = data[categorical_cols[-1]]  # 假设目标变量是最后一列

selector = SelectKBest(mutual_info_classif, k=10)
X_reduced = selector.fit_transform(X, y)

# 4. PCA降维
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_reduced)

# 5. 分层抽样
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
train_idx, test_idx = next(split.split(X, y))
data_reduced = data.iloc[test_idx]

# 6. 可视化
# 特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns[selector.get_support()],
    'Score': selector.scores_[selector.get_support()]
}).sort_values(by='Score', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=feature_importance)
plt.title('Top 10 Feature Importance')
plt.show()

# PCA方差解释率
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_),
         marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Variance Explained')
plt.show()