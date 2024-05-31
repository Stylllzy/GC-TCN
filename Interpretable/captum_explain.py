# -*- coding: utf-8 -*-
import numpy as np
import torch
from captum.attr import FeaturePermutation
from matplotlib import pyplot as plt
import seaborn as sns


def captum_feature_permutation(model, loader, device):
    """
        可解释性分析,计算特征重要性
    """
    print(
        '#--------------------------------------------开始特征可解释性分析---------------------------------------------#')
    print("---------------------------------------Captum Feature Permutation ...")
    feature_perm = FeaturePermutation(model.forward)

    # 示例：使用第一个批次的数据来计算特征重要性
    batch = next(iter(loader)).detach().to(device)
    x, edge_index, target_node_index = batch.x, batch.edge_index, batch.target_node_index

    attributions = feature_perm.attribute(inputs=(x, edge_index, target_node_index), target=None, show_progress=True)
    # 存储 attributions
    torch.save(attributions, 'attributions_test.pt')
    print(type(attributions))
    print(len(attributions))
    print(attributions[0].shape)

    return attributions


def draw_fea(mean_across_cities, feature_labels):
    # 设置 seaborn 的风格
    # sns.set_theme(style="whitegrid")
    max_abs_value = np.max(np.abs(mean_across_cities))

    # 创建一个横向的条形图来显示特征的平均重要性
    plt.figure(figsize=(20, 15))
    ax = sns.barplot(x=mean_across_cities, y=feature_labels)

    # 设置标题和轴标签
    ax.set_xlim(-max_abs_value, max_abs_value)

    ax.set_title('Overall Mean Feature Importance of Each Feature')
    ax.set_xlabel('Overall Mean Feature Importance')
    ax.set_ylabel('Feature Name')

    plt.savefig('feature_importance.png', dpi=300)
    plt.show()


def draw_fea_sorted(mean_across_cities, feature_labels):
    # 计算绝对值的最大值以确保横轴和纵轴的绝对值一致
    max_abs_value = np.max(np.abs(mean_across_cities))

    # 将特征标签和它们对应的平均重要性值结合在一起，以便排序
    features_with_values = list(zip(feature_labels, mean_across_cities))

    # 按照特征重要性的绝对值大小进行排序
    features_with_values.sort(key=lambda x: x[1], reverse=True)

    # 分离排序后的特征标签和它们对应的值
    sorted_labels, sorted_values = zip(*features_with_values)

    # 创建一个横向的条形图来显示排序后的特征的平均重要性
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=np.array(sorted_values), y=np.array(sorted_labels))

    # 设置横轴的范围，使其与纵轴的范围绝对值一致
    ax.set_xlim(-max_abs_value, max_abs_value)

    # 设置标题和轴标签
    ax.set_title('Overall Mean Feature Importance of Each Feature')
    ax.set_xlabel('Overall Mean Feature Importance')
    ax.set_ylabel('Feature Name')

    plt.savefig('feature_importance_sorted.png', dpi=300)
    plt.show()
