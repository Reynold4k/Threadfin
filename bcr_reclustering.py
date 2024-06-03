# threadfin.py
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def bcr_reclustering(adata, bcr_table):
    # 提取 UMAP 坐标
    umap_coords = adata.obsm['X_umap']

    # 提取克隆 ID
    clone_ids = adata.obs['clone_id']

    # 创建 DataFrame，包含 UMAP 坐标和克隆 ID
    umap_df = pd.DataFrame(umap_coords, columns=['umap_x', 'umap_y'])
    umap_df.index = adata.obs.index
    umap_df['clone_id'] = clone_ids
    grouped = umap_df.groupby('clone_id')

    clone_centers = []
    # 遍历每个克隆ID和对应的细胞组
    for clone_id, group in grouped:
        center = group[['umap_x', 'umap_y']].mean()  # 计算坐标的平均值
        # 将克隆ID作为索引添加到DataFrame中
        center.name = clone_id
        clone_centers.append(center)

    clone_adata = pd.DataFrame(clone_centers, columns=['umap_x', 'umap_y'])
    clone_adata_object = sc.AnnData(clone_adata)

    # 初始化一个字典来存储克隆 ID 的比较结果
    clone_comparisons = {}

    # 遍历所有克隆 ID
    for clone_id in adata.obs['clone_id'].unique():
        # 计算当前克隆 ID 对应的参考坐标
        clone_reference = np.mean(adata[adata.obs['clone_id'] == clone_id].obsm['X_umap'], axis=0)
        
        # 获取当前克隆 ID 对应的检查坐标
        clone_checkpoint = clone_adata[clone_adata.index == clone_id].values[0]
        
        # 判断两个数组是否相等，并存储比较结果
        clone_comparisons[clone_id] = np.allclose(clone_reference, clone_checkpoint)
    
    unequal_count = 0

    # 遍历所有克隆 ID
    for clone_id, comparison_result in clone_comparisons.items():
        if not comparison_result:
            unequal_count += 1

    # 输出结果
    if unequal_count == 0:
        print("Clone Coordinates Check: Perfect!! All indexes match!!")
    else:
        print(f"{unequal_count} 个克隆的坐标不相等. Please have a check at the index again")

    # 创建 UMAP 数据框并添加注释
    umap_df = pd.DataFrame({'UMAP 1': umap_coords[:, 0], 'UMAP 2': umap_coords[:, 1]})
    umap_df.index = clone_adata.index  # 使用克隆的索引作为UMAP数据的索引
    split_index = umap_df.index.str.split('_')
    umap_df['v_call'] = split_index.str[0]
    umap_df['d_call'] = split_index.str[1]
    umap_df['j_call'] = split_index.str[2]
    umap_df['v_call'] = umap_df['v_call'].astype('category')
    umap_df['d_call'] = umap_df['d_call'].astype('category')
    umap_df['j_call'] = umap_df['j_call'].astype('category')
    umap_df['np_specific'] = (umap_df['v_call'] == 'IGHV1-72*01').astype(int)

    adata_umap = sc.AnnData(umap_df)

    sc.pp.neighbors(adata_umap)
    sc.tl.leiden(adata_umap, resolution=0.3)
    umap_df['clone_cluster'] = adata_umap.obs['leiden']

    # 定义颜色映射
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                       '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff',
                       '#ff00ff', '#800000', '#008000', '#000080', '#808000']
    unique_clusters = sorted(adata_umap.obs['leiden'].unique())
    cluster_color_mapping = dict(zip(unique_clusters, distinct_colors))

    # 筛选克隆数量，细胞数量体现点大小
    clone_sizes = adata.obs['clone_id'].value_counts().sort_index()
    clone_sizes_filtered = clone_sizes[clone_sizes >= 1]
    umap_df['clone_size'] = umap_df.index.map(clone_sizes_filtered)
    umap_df['clone_size'].fillna(0, inplace=True)

    min_size = 1
    max_size = adata.obs['clone_id'].value_counts()[0]
    umap_df['scaled_clone_size'] = umap_df['clone_size']
    umap_df['scaled_clone_size'] = umap_df['clone_size'].apply(lambda x: (x - 3) if x >= 3 else 0)
    umap_df['scaled_clone_size'] = (umap_df['scaled_clone_size'] - umap_df['scaled_clone_size'].min()) / (umap_df['scaled_clone_size'].max() - umap_df['scaled_clone_size'].min())
    umap_df['scaled_clone_size'] = umap_df['scaled_clone_size'] * (max_size - min_size) + min_size

    sns.scatterplot(data=umap_df, x='UMAP 1', y='UMAP 2', hue='clone_cluster', palette=cluster_color_mapping, size=umap_df['scaled_clone_size'], sizes=(15, max_size))
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=umap_df, x='UMAP 1', y='UMAP 2', hue='np_specific', size=umap_df['scaled_clone_size'], sizes=(15, max_size))
    plt.show()

    adata.obs['clone_cluster'] = adata.obs['clone_id'].map(umap_df['clone_cluster'])
    adata.obs['clone_cluster'] = adata.obs['clone_cluster'].astype('category')

    plt.figure(figsize=(10, 8))  
    sns.scatterplot(x=adata.obsm['X_umap'][:, 0], y=adata.obsm['X_umap'][:, 1], hue=adata.obs['clone_cluster'], palette=cluster_color_mapping, s=50)
    plt.show()

    return adata
