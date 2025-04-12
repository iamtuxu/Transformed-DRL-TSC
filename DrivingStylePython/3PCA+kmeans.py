'''
This code is used for PCA and kmeans clustering.  and plot
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

# Define the CSV file path
csv_file = 'data/train.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file, delimiter=',')


# Group by 'Vehicle_ID' and 'Sequence Index'
grouped_df = df.groupby(['Vehicle_ID', 'Sequence_Index'])

### first step: extract car-following characteristics
data = []
for _, df in grouped_df:
    data.append([df['Ego_V'].max(), df['Ego_V'].mean(), df['Ego_V'].std(), df['Ego_A'].max(),
           df['Ego_A'].min(), df['Ego_A'].mean(), df['Ego_A'].std(), df['Gap'].max(),
          df['Gap'].min(), df['Gap'].mean(), df['Gap'].std(),
          (df['Leader_V'] - df['Ego_V']).mean(), (df['Leader_V'] - df['Ego_V']).std()])
# print(data)

# 标准化
scaler_standard = StandardScaler()
data_standardized = scaler_standard.fit_transform(data)

print("标准化后的数据：")
print(data_standardized)

# 归一化
scaler_minmax = MinMaxScaler()
data_normalized = scaler_minmax.fit_transform(data)

print("\n归一化后的数据：")
print(data_normalized)


pca = PCA(n_components=5)
newData = pca.fit_transform(data_standardized)
print(newData)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

# # elbow method: optimal clusters 4
# wcss = []
# for i in range(1, 10):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(newData)
#     wcss.append(kmeans.inertia_)
#
# # 设置图形大小，宽度为10英寸，高度为5英寸，以达到2:1的比例
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, 10), wcss, marker='+', color='#007bff')  # 使用更深的蓝色
# plt.xlabel('Number of clusters: K')
# plt.ylabel('SSE')
# plt.tight_layout()
# plt.show()
#
# print(wcss)

# 2D
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(newData)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(newData[:,0], newData[:,1], c=pred_y, s=60, cmap='viridis', alpha = 0.8)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='+')
plt.show()
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

unique, counts = np.unique(pred_y, return_counts=True)
cluster_counts = dict(zip(unique, counts))
print("Cluster counts:", cluster_counts)

# Set up a color palette
colors = plt.cm.viridis(np.linspace(0, 1, 4))

# Fit KMeans on the transformed data
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(newData)

# Plotting
fig = plt.figure(figsize=(10, 8))  # Adjusting figure size
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for PC1, PC2, and PC3
for i in range(4):
    ax.scatter(newData[pred_y == i, 0], newData[pred_y == i, 1], newData[pred_y == i, 2], c=[colors[i]], s=80, alpha=0.7, label=f'Cluster {i+1}')  # Using rainbow color palette


# Labeling axes
ax.set_xlabel('PC1', fontsize=14)  # Adjusting font size
ax.set_ylabel('PC2', fontsize=14)
ax.set_zlabel('PC3', fontsize=14)
ax.set_title('3D Visualization of KMeans Clustering', fontsize=16)  # Adding title

# Adding legend
ax.legend()

# Setting axis limits based on data range
x_min, x_max = newData[:, 0].min() - 0.5, newData[:, 0].max() + 0.5
y_min, y_max = newData[:, 1].min() - 0.5, newData[:, 1].max() + 0.5
z_min, z_max = newData[:, 2].min() - 0.5, newData[:, 2].max() + 0.5
ax.set_xlim(-6, 7)
ax.set_ylim(-5, 8)
ax.set_zlim(-2, 5)



plt.show()
# # 创建一个空的DataFrame来存储所有数据
# merged_df = pd.DataFrame()
# i = 0
# for _, df in grouped_df:
#     df['cluster'] = kmeans.labels_[i]
#     merged_df = pd.concat([merged_df, df])
#     i += 1
# # 将合并后的DataFrame保存到CSV文件中
# merged_df.to_csv('data/train_cluster.csv', index=False)