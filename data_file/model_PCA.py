import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import warnings
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib.patches import Ellipse


warnings.filterwarnings('ignore')

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 降维模型 PCA 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"pca_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """PCA参数配置"""
        # 1. 扩展算法参数
        pca_params = {
            "target_column": "target",
            # 目标列名（用于着色）
            # 默认值：'target'
            # 注意：如果数据没有目标列，可以设置为None，则所有点使用相同颜色

            "features_only": False,
            # 是否只使用特征列（排除目标列）
            # 取值范围：布尔值 True 或 False
            #   True: 仅使用特征列进行PCA分析
            #   False: 包含所有列，但目标列仅用于着色
            # 默认值：False

            "standardize": True,
            # 是否标准化特征
            # 取值范围：布尔值 True 或 False
            #   True: 推荐，因为PCA对数据尺度敏感
            #   False: 使用原始数据尺度
            # 默认值：True

            "n_components": None,
            # 主成分数量
            # 取值范围：
            #   None: 保留所有主成分
            #   整数 (如 2, 3, 5): 保留指定数量的主成分
            #   浮点数 (如 0.95, 0.80): 保留累计解释方差达到该比例的主成分
            #   'mle': 使用MLE（最大似然估计）自动选择主成分数量
            # 默认值：None

            "whiten": False,
            # 是否白化（Whitening）
            # 取值范围：布尔值 True 或 False
            #   True: 白化数据，使各主成分具有单位方差
            #   False: 保留原始尺度（默认）
            # 注意：白化通常用于某些算法（如K-means）的预处理
            # 默认值：False

            "svd_solver": "auto",
            # SVD求解器
            # 可选值：
            #   'auto': 自动选择（默认），根据数据大小和n_components参数自动选择
            #   'full': 完整的SVD（LAPACK），适用于所有情况但计算成本高
            #   'arpack': 截断SVD，适用于n_components较小的情况
            #   'randomized': 随机SVD，适用于大数据集，牺牲精度换取速度
            # 默认值：'auto'

            "random_state": None,
            # 随机种子
            # 取值范围：正整数或None
            #   None: 非确定性结果
            #   整数: 确保结果可重现
            # 默认值：None

            "tol": 0.0,
            # 奇异值的容忍度
            # 取值范围：大于等于0的浮点数
            #   当svd_solver='arpack'时使用
            # 默认值：0.0

            "iterated_power": "auto",
            # 幂方法的迭代次数
            # 取值范围：
            #   'auto': 自动选择
            #   整数: 指定迭代次数
            # 默认值：'auto'

            "variance_threshold": 0.95,
            # 方差解释阈值
            # 取值范围：0到1之间的浮点数
            #   用于标记累计解释方差达到该阈值的主成分数量
            # 默认值：0.95

            "biplot": False,
            # 是否生成双标图（Biplot）
            # 取值范围：布尔值 True 或 False
            #   True: 生成双标图，同时显示样本和特征向量
            #   False: 仅显示样本点
            # 默认值：False

            "ellipse_confidence": 0.95,
            # 置信椭圆置信水平
            # 取值范围：0到1之间的浮点数
            #   为每个类别绘制置信椭圆
            # 默认值：0.95

            "color_by_target": True,
            # 是否按目标变量着色
            # 取值范围：布尔值 True 或 False
            #   True: 按目标变量类别着色
            #   False: 所有点使用相同颜色
            # 默认值：True
        }

        # 从传入参数中更新PCA参数
        in_params = params.get("params", {})

        # 更新参数 - 处理可能的列表值
        if "target_column" in in_params:
            pca_params["target_column"] = in_params["target_column"]

        if "features_only" in in_params:
            pca_params["features_only"] = in_params["features_only"]

        if "standardize" in in_params:
            pca_params["standardize"] = in_params["standardize"]

        # n_components参数处理
        if "n_components" in in_params:
            n_comp_val = in_params["n_components"]
            if n_comp_val == "None":
                pca_params["n_components"] = None
            elif n_comp_val == "mle":
                pca_params["n_components"] = 'mle'
            elif n_comp_val == "3":
                pca_params["n_components"] = 3
            elif n_comp_val == "0.95":
                pca_params["n_components"] = 0.95
            else:
                # 尝试转换为数字
                try:
                    if '.' in str(n_comp_val):
                        pca_params["n_components"] = float(n_comp_val)
                    else:
                        pca_params["n_components"] = int(n_comp_val)
                except:
                    pca_params["n_components"] = None

        if "whiten" in in_params:
            whiten_val = in_params["whiten"]
            if isinstance(whiten_val, str):
                pca_params["whiten"] = (whiten_val.lower() == "true")
            else:
                pca_params["whiten"] = bool(whiten_val)

        if "svd_solver" in in_params:
            svd_val = in_params["svd_solver"]
            if svd_val in ["auto", "full", "arpack", "randomized"]:
                pca_params["svd_solver"] = svd_val

        if "random_state" in in_params:
            random_val = in_params["random_state"]
            pca_params["random_state"] = AlgModelParameters.format_to_int(random_val)

        if "tol" in in_params:
            tol_val = in_params["tol"]
            pca_params["tol"] = AlgModelParameters.format_to_float(tol_val)

        if "iterated_power" in in_params:
            iter_val = in_params["iterated_power"]
            if iter_val == "auto":
                pca_params["iterated_power"] = 'auto'
            else:
                pca_params["iterated_power"] = AlgModelParameters.format_to_int(iter_val)

        if "variance_threshold" in in_params:
            var_val = in_params["variance_threshold"]
            pca_params["variance_threshold"] = AlgModelParameters.format_to_float(var_val)

        if "biplot" in in_params:
            biplot_val = in_params["biplot"]
            if isinstance(biplot_val, str):
                pca_params["biplot"] = (biplot_val.lower() == "true")
            else:
                pca_params["biplot"] = bool(biplot_val)

        if "ellipse_confidence" in in_params:
            ellipse_val = in_params["ellipse_confidence"]
            pca_params["ellipse_confidence"] = AlgModelParameters.format_to_float(ellipse_val)

        if "color_by_target" in in_params:
            color_val = in_params["color_by_target"]
            if isinstance(color_val, str):
                pca_params["color_by_target"] = (color_val.lower() == "true")
            else:
                pca_params["color_by_target"] = bool(color_val)

        # 打印参数信息
        print(f"\nPCA算法参数配置:")
        print(f"  目标列: {pca_params['target_column']}")
        print(f"  仅特征列: {pca_params['features_only']}")
        print(f"  标准化: {pca_params['standardize']}")
        print(f"  主成分数量: {pca_params['n_components']}")
        print(f"  白化: {pca_params['whiten']}")
        print(f"  SVD求解器: {pca_params['svd_solver']}")
        print(f"  随机种子: {pca_params['random_state']}")
        print(f"  方差解释阈值: {pca_params['variance_threshold']}")
        print(f"  双标图: {pca_params['biplot']}")
        print(f"  置信椭圆: {pca_params['ellipse_confidence']}")
        print(f"  按目标着色: {pca_params['color_by_target']}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("降维模型", "PCA").copy()

        # 如果用户提供了自定义图片参数，则更新
        if "image_param" in params and isinstance(params["image_param"], dict):
            plot_config.update(params["image_param"])

        # 3. 数据准备
        # 确定目标列
        y_column = pca_params["target_column"]

        # 检查目标列是否存在
        y_exists = y_column in df.columns

        if pca_params["features_only"] or not y_exists:
            # 仅使用特征列或目标列不存在
            X_columns = [col for col in df.columns if col != "target"]
            X = df[X_columns]
            y = None
            print(f"\n仅使用特征列进行PCA分析")
            print(f"  特征数量: {len(X_columns)}")
        else:
            # 使用目标列进行着色
            X_columns = [col for col in df.columns if col != y_column]
            X = df[X_columns]
            y = df[y_column]
            print(f"\n使用目标列 '{y_column}' 进行着色")
            print(f"  特征数量: {len(X_columns)}")
            print(f"  目标变量类型: {y.dtype}")

            # 如果目标是数值型，可以将其转换为类别用于着色
            if y.dtype in ['int64', 'float64']:
                print(f"  目标变量范围: {y.min():.4f} - {y.max():.4f}")
                # 将数值型目标分箱为类别（如果值太多）
                if len(y.unique()) > 20:
                    print("目标变量有太多唯一值，将进行分箱处理")
                    y = pd.cut(y, bins=10, labels=False)
                    y = y.astype(str) + "_bin"

        print(f"\n数据信息:")
        print(f"  总样本数: {df.shape[0]}")
        print(f"  特征数据形状: {X.shape}")
        if y is not None:
            print(f"  目标数据形状: {y.shape}")
            print(f"  目标变量唯一值数量: {len(y.unique())}")

        # 移除包含NaN的行
        if X.isna().any().any():
            print(f"警告: 特征数据中包含NaN值，正在移除...")
            X_clean = X.dropna()
            if y is not None:
                y_clean = y[X.index.isin(X_clean.index)]
            else:
                y_clean = None
            print(f"  移除后样本数: {X_clean.shape[0]}")
            X = X_clean
            y = y_clean

        # 4. 数据标准化
        if pca_params["standardize"]:
            print("\n标准化特征数据...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # 5. 应用PCA
        print(f"\n应用PCA降维...")

        # 准备PCA参数
        pca_kwargs = {
            "n_components": pca_params["n_components"],
            "whiten": pca_params["whiten"],
            "svd_solver": pca_params["svd_solver"],
            "random_state": pca_params["random_state"],
            "tol": pca_params["tol"],
            "iterated_power": pca_params["iterated_power"]
        }

        # 移除None值参数
        pca_kwargs = {k: v for k, v in pca_kwargs.items() if v is not None}

        pca = PCA(**pca_kwargs)
        X_pca = pca.fit_transform(X_scaled)

        # 6. PCA结果分析
        n_components = pca.n_components_
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = pca.explained_variance_
        components = pca.components_

        # 计算累计解释方差
        cumulative_variance = np.cumsum(explained_variance_ratio)

        print(f"\nPCA结果:")
        print(f"  主成分数量: {n_components}")
        print(f"  总解释方差比例: {np.sum(explained_variance_ratio):.4f}")

        # 找到达到方差阈值的主成分数量
        threshold = pca_params["variance_threshold"]
        n_components_threshold = np.where(cumulative_variance >= threshold)[0]
        if len(n_components_threshold) > 0:
            n_components_for_threshold = n_components_threshold[0] + 1
            print(f"  达到{threshold * 100}%方差所需主成分: {n_components_for_threshold}")
        else:
            n_components_for_threshold = n_components
            print(f"  警告: 所有主成分累计方差未达到{threshold * 100}%阈值")

        # 7. 生成碎石图
        plt.figure(figsize=(14, 6))

        # 子图1：碎石图（方差解释比例）
        plt.subplot(1, 2, 1)
        components_idx = range(1, n_components + 1)

        # 使用双Y轴
        ax1 = plt.gca()
        bars = ax1.bar(components_idx, explained_variance_ratio, alpha=0.7, color='steelblue', label='方差解释比例')
        ax1.set_xlabel(plot_config["scree_plot_xlabel"], fontsize=12)
        ax1.set_ylabel(plot_config["scree_plot_ylabel"], fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_title(plot_config["scree_plot_title"], fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值标签
        for bar, val in zip(bars, explained_variance_ratio):
            height = bar.get_height()
            if val > 0.01:  # 只显示大于1%的值
                ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{val:.1%}', ha='center', va='bottom', fontsize=9)

        # 子图2：累计解释方差比例
        plt.subplot(1, 2, 2)
        plt.plot(components_idx, cumulative_variance, 'o-', linewidth=2, markersize=8, color='darkorange',
                 label='累计方差')
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.6, label=f'{threshold * 100}%阈值')

        # 标记达到阈值的主成分
        if len(n_components_threshold) > 0:
            plt.axvline(x=n_components_for_threshold, color='g', linestyle=':', alpha=0.6)
            plt.text(n_components_for_threshold, threshold / 2,
                     f'主成分{n_components_for_threshold}\n达到{threshold * 100}%',
                     ha='right', va='center', fontsize=9, color='g')

        plt.xlabel(plot_config["cumulative_variance_xlabel"], fontsize=12)
        plt.ylabel(plot_config["cumulative_variance_ylabel"], fontsize=12)
        plt.title(plot_config["cumulative_variance_title"], fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        scree_plot_path = os.path.join(output_dir, 'scree_plot.png')
        plt.savefig(scree_plot_path, dpi=dpi)
        plt.close()

        # 保存碎石图数据
        scree_data_path = os.path.join(output_dir, 'scree_plot_data.txt')
        scree_data = pd.DataFrame({
            'component': components_idx,
            'explained_variance': explained_variance,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance
        })
        scree_data.to_csv(scree_data_path, sep='\t', index=False)

        imgs.append({
            "name": "PCA碎石图",
            "img": scree_plot_path,
            "data": scree_data_path
        })

        # 8. 生成二维散点图
        if n_components >= 2:
            plt.figure(figsize=(12, 10))

            # 准备颜色映射
            if y is not None and pca_params["color_by_target"]:
                unique_labels = y.unique()
                n_labels = len(unique_labels)
                colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
                color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
                point_colors = [color_map[label] for label in y]
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=color_map[label],
                                              markersize=10, label=str(label))
                                   for label in unique_labels]
            else:
                point_colors = 'steelblue'
                legend_elements = []

            # 绘制散点图
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=point_colors, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)

            # 添加置信椭圆
            if y is not None and pca_params["color_by_target"] and pca_params["ellipse_confidence"] > 0:
                for label in unique_labels:
                    mask = (y == label)
                    if np.sum(mask) > 2:  # 至少需要3个点来绘制椭圆
                        x_vals = X_pca[mask, 0]
                        y_vals = X_pca[mask, 1]

                        # 计算椭圆的参数
                        cov = np.cov(x_vals, y_vals)
                        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

                        # 使用卡方分布获取椭圆半径
                        ell_radius_x = np.sqrt(1 + pearson) * np.sqrt(cov[0, 0]) * 2
                        ell_radius_y = np.sqrt(1 - pearson) * np.sqrt(cov[1, 1]) * 2

                        ellipse = Ellipse((np.mean(x_vals), np.mean(y_vals)),
                                          width=ell_radius_x, height=ell_radius_y,
                                          alpha=0.2, color=color_map[label])
                        plt.gca().add_patch(ellipse)

            # 添加标签和标题
            pc1_var = explained_variance_ratio[0] * 100
            pc2_var = explained_variance_ratio[1] * 100
            plt.xlabel(f"{plot_config['pca_2d_scatter_xlabel']} ({pc1_var:.1f}%)", fontsize=12)
            plt.ylabel(f"{plot_config['pca_2d_scatter_ylabel']} ({pc2_var:.1f}%)", fontsize=12)
            plt.title(f"{plot_config['pca_2d_scatter_title']}\n累计解释方差: {cumulative_variance[1]:.1%}",
                      fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # 添加图例
            if legend_elements:
                plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

            # 添加解释文本
            text_str = f'样本数: {X_pca.shape[0]}\n特征数: {X.shape[1]}\n主成分数: {n_components}'
            plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            scatter_2d_path = os.path.join(output_dir, 'pca_2d_scatter.png')
            plt.savefig(scatter_2d_path, dpi=dpi)
            plt.close()

            # 保存二维散点图数据
            scatter_2d_data_path = os.path.join(output_dir, 'pca_2d_scatter_data.txt')
            scatter_2d_data = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'target': y if y is not None else ['N/A'] * len(X_pca)
            })
            scatter_2d_data.to_csv(scatter_2d_data_path, sep='\t', index=False)

            imgs.append({
                "name": "PCA二维散点图",
                "img": scatter_2d_path,
                "data": scatter_2d_data_path
            })

        # 9. 生成三维散点图（如果主成分数量>=3）
        if n_components >= 3:
            try:
                from mpl_toolkits.mplot3d import Axes3D

                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')

                # 准备颜色映射
                if y is not None and pca_params["color_by_target"]:
                    unique_labels = y.unique()
                    n_labels = len(unique_labels)
                    colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
                    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
                    point_colors = [color_map[label] for label in y]
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=color_map[label],
                                                  markersize=10, label=str(label))
                                       for label in unique_labels[:10]]  # 只显示前10个图例
                else:
                    point_colors = 'steelblue'
                    legend_elements = []

                # 绘制三维散点图
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                                     c=point_colors, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)

                # 添加标签和标题
                pc1_var = explained_variance_ratio[0] * 100
                pc2_var = explained_variance_ratio[1] * 100
                pc3_var = explained_variance_ratio[2] * 100
                ax.set_xlabel(f"{plot_config['pca_3d_scatter_xlabel']} ({pc1_var:.1f}%)", fontsize=11)
                ax.set_ylabel(f"{plot_config['pca_3d_scatter_ylabel']} ({pc2_var:.1f}%)", fontsize=11)
                ax.set_zlabel(f"{plot_config['pca_3d_scatter_zlabel']} ({pc3_var:.1f}%)", fontsize=11)
                ax.set_title(f"{plot_config['pca_3d_scatter_title']}\n累计解释方差: {cumulative_variance[2]:.1%}",
                             fontsize=14, fontweight='bold')

                # 添加图例
                if legend_elements:
                    ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left')

                plt.tight_layout()
                scatter_3d_path = os.path.join(output_dir, 'pca_3d_scatter.png')
                plt.savefig(scatter_3d_path, dpi=dpi)
                plt.close()

                # 保存三维散点图数据
                scatter_3d_data_path = os.path.join(output_dir, 'pca_3d_scatter_data.txt')
                scatter_3d_data = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'PC3': X_pca[:, 2],
                    'target': y if y is not None else ['N/A'] * len(X_pca)
                })
                scatter_3d_data.to_csv(scatter_3d_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "PCA三维散点图",
                    "img": scatter_3d_path,
                    "data": scatter_3d_data_path
                })
            except Exception as e:
                print(f"生成三维散点图时出错: {e}")

        # 10. 生成双标图（Biplot，如果主成分数量>=2）
        if n_components >= 2 and pca_params["biplot"]:
            plt.figure(figsize=(12, 10))

            # 绘制样本点
            if y is not None and pca_params["color_by_target"]:
                unique_labels = y.unique()
                n_labels = len(unique_labels)
                colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
                color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
                point_colors = [color_map[label] for label in y]
            else:
                point_colors = 'steelblue'

            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=point_colors, alpha=0.5, s=30)

            # 绘制特征向量（载荷）
            feature_vectors = components[:2, :].T  # 转置以匹配形状
            scale_factor = 5  # 缩放因子，使向量更可见

            for i, (vec_x, vec_y) in enumerate(feature_vectors):
                plt.arrow(0, 0, vec_x * scale_factor, vec_y * scale_factor,
                          head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.7)
                plt.text(vec_x * scale_factor * 1.15, vec_y * scale_factor * 1.15,
                         X_columns[i] if i < len(X_columns) else f"Feature{i}",
                         color='red', fontsize=9, ha='center', va='center')

            # 添加标签和标题
            pc1_var = explained_variance_ratio[0] * 100
            pc2_var = explained_variance_ratio[1] * 100
            plt.xlabel(f"{plot_config['biplot_xlabel']} ({pc1_var:.1f}%)", fontsize=12)
            plt.ylabel(f"{plot_config['biplot_ylabel']} ({pc2_var:.1f}%)", fontsize=12)
            plt.title(f"{plot_config['biplot_title']}\n红色箭头表示特征向量方向",
                      fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

            plt.tight_layout()
            biplot_path = os.path.join(output_dir, 'pca_biplot.png')
            plt.savefig(biplot_path, dpi=dpi)
            plt.close()

            # 保存双标图数据
            biplot_data_path = os.path.join(output_dir, 'pca_biplot_data.txt')
            biplot_data = pd.DataFrame({
                'feature': X_columns,
                'PC1_loading': components[0, :],
                'PC2_loading': components[1, :]
            })
            biplot_data.to_csv(biplot_data_path, sep='\t', index=False)

            imgs.append({
                "name": "PCA双标图",
                "img": biplot_path,
                "data": biplot_data_path
            })

        # 11. 生成主成分载荷图（前10个特征对前3个主成分的贡献）
        if n_components >= 3:
            plt.figure(figsize=(14, 8))

            n_features_to_show = min(15, len(X_columns))
            top_features_idx = np.argsort(np.abs(components[0, :]))[-n_features_to_show:]
            top_features = [X_columns[i] for i in top_features_idx]

            # 为前3个主成分创建分组柱状图
            bar_width = 0.25
            index = np.arange(n_features_to_show)

            plt.bar(index - bar_width, components[0, top_features_idx], bar_width,
                    label=f'PC1 ({explained_variance_ratio[0] * 100:.1f}%)', alpha=0.8)
            plt.bar(index, components[1, top_features_idx], bar_width,
                    label=f'PC2 ({explained_variance_ratio[1] * 100:.1f}%)', alpha=0.8)
            plt.bar(index + bar_width, components[2, top_features_idx], bar_width,
                    label=f'PC3 ({explained_variance_ratio[2] * 100:.1f}%)', alpha=0.8)

            plt.xlabel(plot_config["loading_plot_xlabel"], fontsize=12)
            plt.ylabel(plot_config["loading_plot_ylabel"], fontsize=12)
            plt.title(f"{plot_config['loading_plot_title']}\n前{n_features_to_show}个特征对前3个主成分的贡献",
                      fontsize=14, fontweight='bold')
            plt.xticks(index, top_features, rotation=45, ha='right', fontsize=10)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            loading_plot_path = os.path.join(output_dir, 'pca_loading_plot.png')
            plt.savefig(loading_plot_path, dpi=dpi)
            plt.close()

            # 保存载荷数据
            loading_data_path = os.path.join(output_dir, 'pca_loading_data.txt')
            loading_data = pd.DataFrame(components[:n_components, :].T,
                                        columns=[f'PC{i + 1}' for i in range(n_components)],
                                        index=X_columns)
            loading_data.to_csv(loading_data_path, sep='\t')

            imgs.append({
                "name": "主成分载荷图",
                "img": loading_plot_path,
                "data": loading_data_path
            })

        # 12. 生成主成分热图
        if n_components >= 2 and len(X_columns) <= 50:  # 限制特征数量，避免热图过大
            plt.figure(figsize=(14, 10))

            # 选择前20个最重要的特征（按PC1的绝对值）
            n_features_heatmap = min(20, len(X_columns))
            feature_importance = np.abs(components[0, :])
            top_features_idx = np.argsort(feature_importance)[-n_features_heatmap:]
            top_features = [X_columns[i] for i in top_features_idx]

            # 创建热图数据
            heatmap_data = components[:min(10, n_components), top_features_idx].T

            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        xticklabels=[f'PC{i + 1}' for i in range(min(10, n_components))],
                        yticklabels=top_features, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

            plt.xlabel(plot_config["heatmap_ylabel"], fontsize=12)
            plt.ylabel(plot_config["heatmap_xlabel"], fontsize=12)
            plt.title(
                f"{plot_config['heatmap_title']}\n前{n_features_heatmap}个特征对前{min(10, n_components)}个主成分的载荷",
                fontsize=14, fontweight='bold')
            plt.tight_layout()

            heatmap_path = os.path.join(output_dir, 'pca_heatmap.png')
            plt.savefig(heatmap_path, dpi=dpi)
            plt.close()

            imgs.append({
                "name": "PCA主成分热图",
                "img": heatmap_path,
                "data": loading_data_path  # 使用相同的载荷数据
            })

        # 13. 生成对图（Pairplot，如果主成分数量>=4）
        if n_components >= 4 and X_pca.shape[0] <= 1000:  # 限制样本数量，避免图形过大
            try:
                # 选择前4个主成分
                pairplot_data = pd.DataFrame(X_pca[:, :4], columns=[f'PC{i + 1}' for i in range(4)])
                if y is not None:
                    pairplot_data['target'] = y

                # 创建对图
                pairplot = sns.pairplot(pairplot_data, hue='target' if y is not None else None,
                                        diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
                pairplot.fig.suptitle(f"{plot_config['pairplot_title']}\n前4个主成分的分布关系",
                                      fontsize=16, fontweight='bold', y=1.02)

                pairplot_path = os.path.join(output_dir, 'pca_pairplot.png')
                pairplot.fig.tight_layout()
                pairplot.fig.savefig(pairplot_path, dpi=dpi)
                plt.close(pairplot.fig)

                imgs.append({
                    "name": "PCA对图",
                    "img": pairplot_path,
                    "data": scatter_2d_data_path  # 使用二维散点图数据
                })
            except Exception as e:
                print(f"生成对图时出错: {e}")

        # 14. 保存PCA结果数据
        # 保存主成分得分
        pca_scores_path = os.path.join(output_dir, 'pca_scores.txt')
        pca_scores_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
        if y is not None:
            pca_scores_df['target'] = y
        pca_scores_df.to_csv(pca_scores_path, sep='\t', index=False)

        # 15. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": X.shape[0],
                "features": X.shape[1],
                "target_variable": y_column if y_exists else None,
                "feature_variables": X_columns,
                "target_used": y is not None
            },
            "model_params": pca_params,
            "pca_results": {
                "n_components": int(n_components),
                "total_explained_variance": float(np.sum(explained_variance_ratio)),
                "explained_variance_ratio": [float(v) for v in explained_variance_ratio],
                "explained_variance": [float(v) for v in explained_variance],
                "cumulative_variance": [float(v) for v in cumulative_variance],
                "components_for_threshold": int(
                    n_components_for_threshold) if 'n_components_for_threshold' in locals() else n_components
            },
            "files": {
                "pca_scores": os.path.basename(pca_scores_path),
                "scree_data": os.path.basename(scree_data_path) if 'scree_data_path' in locals() else None,
                "loading_data": os.path.basename(loading_data_path) if 'loading_data_path' in locals() else None
            }
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 16. 生成结果字符串
        result_str += "=======================\n"
        result_str += "PCA模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征\n"
        result_str += f"2. 目标变量: {y_column if y_exists else '无'}\n"
        result_str += f"3. 主成分数量: {n_components}\n"
        result_str += f"4. 总解释方差比例: {np.sum(explained_variance_ratio):.4f}\n"

        # 显示前几个主成分的解释方差
        result_str += f"\n=== 前5个主成分的解释方差 ===\n"
        for i in range(min(5, n_components)):
            result_str += f"  PC{i + 1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i] * 100:.2f}%)\n"

        result_str += f"\n=== 累计解释方差 ===\n"
        for i in [1, 2, 3, 5, 10]:
            if i <= n_components:
                result_str += f"  前{i}个主成分: {cumulative_variance[i - 1]:.4f} ({cumulative_variance[i - 1] * 100:.2f}%)\n"

        result_str += f"\n达到{pca_params['variance_threshold'] * 100}%方差所需主成分: {n_components_for_threshold}\n"

        # 添加模型性能评估
        if np.sum(explained_variance_ratio) > 0.9:
            result_str += "5. 降维效果: 优秀 (总解释方差 > 90%)\n"
        elif np.sum(explained_variance_ratio) > 0.7:
            result_str += "5. 降维效果: 良好 (70% < 总解释方差 ≤ 90%)\n"
        elif np.sum(explained_variance_ratio) > 0.5:
            result_str += "5. 降维效果: 一般 (50% < 总解释方差 ≤ 70%)\n"
        else:
            result_str += "5. 降维效果: 较差 (总解释方差 ≤ 50%)\n"

        # 添加生成的图片信息
        result_str += f"\n=== 生成的图片 ===\n"
        for img_info in imgs:
            img_filename = os.path.basename(img_info['img'])
            data_filename = os.path.basename(img_info['data']) if img_info['data'] else "无数据文件"
            result_str += f"- {img_info['name']}: {img_filename} (数据文件: {data_filename})\n"

        # 添加模型信息文件名
        result_str += f"\n=== 模型信息文件 ===\n"
        model_info_filename = os.path.basename(model_info_path) if 'model_info_path' in locals() else "未生成"
        result_str += f"模型信息文件: {model_info_filename}\n"
        result_str += f"主成分得分文件: {os.path.basename(pca_scores_path)}\n"

        # 添加计算使用的参数信息
        result_str += f"\n=== 计算使用的参数信息 ===\n"
        result_str += f"PCA算法参数:\n"

        param_descriptions = {
            "target_column": f"目标列名 (当前: {pca_params['target_column']})",
            "features_only": f"是否只使用特征列 (当前: {pca_params['features_only']})",
            "standardize": f"是否标准化特征 (当前: {pca_params['standardize']})",
            "n_components": f"主成分数量 (当前: {pca_params['n_components']})",
            "whiten": f"是否白化数据 (当前: {pca_params['whiten']})",
            "svd_solver": f"SVD求解器 (当前: {pca_params['svd_solver']})",
            "random_state": f"随机种子 (当前: {pca_params['random_state']})",
            "variance_threshold": f"方差解释阈值 (当前: {pca_params['variance_threshold']})",
            "biplot": f"是否生成双标图 (当前: {pca_params['biplot']})",
            "ellipse_confidence": f"置信椭圆置信水平 (当前: {pca_params['ellipse_confidence']})",
            "color_by_target": f"是否按目标着色 (当前: {pca_params['color_by_target']})"
        }

        for key, value in pca_params.items():
            if key in param_descriptions:
                result_str += f"  {key}: {value}\n"
                result_str += f"    说明: {param_descriptions[key]}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"PCA模型运行失败: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        result_str = "=======================\n"
        result_str += "PCA模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 降维模型 PCA 运行结束 <<<<<<<<<<<<<<<<<<<<")
    print(f"计算耗时: {expend_time} 毫秒")

    result_str += f"\n计算耗时(毫秒):{expend_time}\n"

    # 构建JSON格式的结果
    result_json = {
        "success": True if "运行状态: 成功" in result_str else False,
        "error_msg": "" if "运行状态: 成功" in result_str else result_str.split("错误信息: ")[-1].strip(),
        "summary": result_str,
        "generated_files": {
            "imgs": imgs,
            "model_info": {
                "data": model_info_path if 'model_info_path' in locals() else ""
            },
            "pca_scores": {
                "data": pca_scores_path if 'pca_scores_path' in locals() else ""
            }
        },
        "start": start_timestamp,
        "end": end_timestamp,
        "expend": expend_time
    }

    # 将结果转换为JSON字符串返回
    return json.dumps(result_json, ensure_ascii=False, indent=2)