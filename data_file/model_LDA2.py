import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import warnings
from datetime import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import mahalanobis
from scipy.stats import f_oneway

warnings.filterwarnings('ignore')

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    绘制置信椭圆
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x和y的大小必须相同")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # 使用卡方分布获取椭圆半径
    ell_radius_x = np.sqrt(1 + pearson) * np.sqrt(cov[0, 0]) * n_std
    ell_radius_y = np.sqrt(1 - pearson) * np.sqrt(cov[1, 1]) * n_std

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # 计算均值并平移椭圆
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 降维模型 LDA 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"lda_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """LDA参数配置"""
        # 1. 扩展算法参数
        lda_params = {
            "target_column": "target",
            # 目标列名（必须提供，LDA是监督降维方法）
            # 默认值：'target'
            # 注意：LDA必须有目标变量，且目标变量应该是类别型

            "features_only": True,
            # 是否只使用特征列（排除目标列）
            # 取值范围：布尔值 True 或 False
            #   True: 仅使用特征列进行LDA分析
            #   False: 包含所有列，但目标列仅用于监督降维
            # 默认值：True

            "standardize": True,
            # 是否标准化特征
            # 取值范围：布尔值 True 或 False
            #   True: 推荐，因为LDA对数据尺度敏感
            #   False: 使用原始数据尺度
            # 默认值：True

            "n_components": None,
            # 目标维度（降维后的维度）
            # 取值范围：
            #   None: 自动设置为 min(n_features, n_classes-1)
            #   整数: 指定降维后的维度，不能超过 min(n_features, n_classes-1)
            # 默认值：None

            "solver": "svd",
            # 求解器
            # 可选值：
            #   'svd': 奇异值分解（默认），无需计算协方差矩阵，推荐
            #   'lsqr': 最小二乘解，可用于降维
            #   'eigen': 特征值分解，适用于分类任务
            # 注意：对于降维任务，'svd'通常是最佳选择
            # 默认值：'svd'

            "shrinkage": None,
            # 收缩参数
            # 取值范围：
            #   None: 不收缩（默认）
            #   'auto': 自动Ledoit-Wolf收缩
            #   浮点数: 手动指定收缩系数（0到1之间）
            # 默认值：None

            "priors": None,
            # 各类别的先验概率
            # 取值范围：
            #   None: 使用样本中的类别比例（默认）
            #   数组: 指定各类别的先验概率
            # 默认值：None

            "store_covariance": False,
            # 是否存储类协方差矩阵
            # 取值范围：布尔值 True 或 False
            #   True: 存储协方差矩阵，可用于分析
            #   False: 不存储（默认）
            # 默认值：False

            "tol": 0.0001,
            # 求解器的容忍度
            # 取值范围：大于0的浮点数
            # 默认值：0.0001

            "random_state": None,
            # 随机种子（仅当solver='lsqr'或'eigen'时有效）
            # 取值范围：正整数或None
            # 默认值：None

            "ellipse_confidence": 0.95,
            # 置信椭圆置信水平
            # 取值范围：0到1之间的浮点数
            #   为每个类别绘制置信椭圆
            # 默认值：0.95

            "color_by_target": True,
            # 是否按目标变量着色
            # 取值范围：布尔值 True 或 False
            #   True: 按目标变量类别着色（默认）
            #   False: 所有点使用相同颜色
            # 默认值：True

            "analyze_separation": True,
            # 是否分析类别分离度
            # 取值范围：布尔值 True 或 False
            #   True: 分析LDA后的类别分离效果
            #   False: 仅进行降维
            # 默认值：True

            "visualize_discriminants": True,
            # 是否可视化判别向量
            # 取值范围：布尔值 True 或 False
            #   True: 可视化LDA的判别向量
            #   False: 不可视化
            # 默认值：True

            "calculate_fisher_score": True,
            # 是否计算Fisher得分
            # 取值范围：布尔值 True 或 False
            #   True: 计算每个特征的Fisher判别得分
            #   False: 不计算
            # 默认值：True
        }

        # 从传入参数中更新LDA参数
        in_params = params.get("params", {})

        # 更新参数 - 处理可能的列表值
        if "target_column" in in_params:
            lda_params["target_column"] = in_params["target_column"]

        if "features_only" in in_params:
            lda_params["features_only"] = in_params["features_only"]

        if "standardize" in in_params:
            lda_params["standardize"] = in_params["standardize"]

        # n_components参数处理
        if "n_components" in in_params:
            n_comp_val = in_params["n_components"]
            if n_comp_val == "None":
                lda_params["n_components"] = None
            else:
                lda_params["n_components"] = AlgModelParameters.format_to_int(n_comp_val)

        # solver参数处理
        if "solver" in in_params:
            solver_val = in_params["solver"]
            if solver_val in ["svd", "lsqr", "eigen"]:
                lda_params["solver"] = solver_val

        # shrinkage参数处理
        if "shrinkage" in in_params:
            shrinkage_val = in_params["shrinkage"]
            if shrinkage_val == "None":
                lda_params["shrinkage"] = None
            elif shrinkage_val == "auto":
                lda_params["shrinkage"] = 'auto'
            else:
                lda_params["shrinkage"] = AlgModelParameters.format_to_float(shrinkage_val)

        # priors参数处理
        if "priors" in in_params:
            priors_val = in_params["priors"]
            if priors_val == "None":
                lda_params["priors"] = None

        # store_covariance参数处理
        if "store_covariance" in in_params:
            store_val = in_params["store_covariance"]
            if isinstance(store_val, str):
                lda_params["store_covariance"] = (store_val.lower() == "true")
            else:
                lda_params["store_covariance"] = bool(store_val)

        # tol参数处理
        if "tol" in in_params:
            tol_val = in_params["tol"]
            lda_params["tol"] = AlgModelParameters.format_to_float(tol_val)

        # random_state参数处理
        if "random_state" in in_params:
            random_val = in_params["random_state"]
            lda_params["random_state"] = AlgModelParameters.format_to_int(random_val)

        # ellipse_confidence参数处理
        if "ellipse_confidence" in in_params:
            ellipse_val = in_params["ellipse_confidence"]
            lda_params["ellipse_confidence"] = AlgModelParameters.format_to_float(ellipse_val)

        # color_by_target参数处理
        if "color_by_target" in in_params:
            color_val = in_params["color_by_target"]
            if isinstance(color_val, str):
                lda_params["color_by_target"] = (color_val.lower() == "true")
            else:
                lda_params["color_by_target"] = bool(color_val)

        # analyze_separation参数处理
        if "analyze_separation" in in_params:
            analyze_val = in_params["analyze_separation"]
            if isinstance(analyze_val, str):
                lda_params["analyze_separation"] = (analyze_val.lower() == "true")
            else:
                lda_params["analyze_separation"] = bool(analyze_val)

        # visualize_discriminants参数处理
        if "visualize_discriminants" in in_params:
            visualize_val = in_params["visualize_discriminants"]
            if isinstance(visualize_val, str):
                lda_params["visualize_discriminants"] = (visualize_val.lower() == "true")
            else:
                lda_params["visualize_discriminants"] = bool(visualize_val)

        # calculate_fisher_score参数处理
        if "calculate_fisher_score" in in_params:
            fisher_val = in_params["calculate_fisher_score"]
            if isinstance(fisher_val, str):
                lda_params["calculate_fisher_score"] = (fisher_val.lower() == "true")
            else:
                lda_params["calculate_fisher_score"] = bool(fisher_val)

        # 打印参数信息
        print(f"\nLDA算法参数配置:")
        print(f"  目标列: {lda_params['target_column']}")
        print(f"  仅特征列: {lda_params['features_only']}")
        print(f"  标准化: {lda_params['standardize']}")
        print(f"  目标维度: {lda_params['n_components']}")
        print(f"  求解器: {lda_params['solver']}")
        print(f"  收缩参数: {lda_params['shrinkage']}")
        print(f"  先验概率: {lda_params['priors']}")
        print(f"  存储协方差: {lda_params['store_covariance']}")
        print(f"  随机种子: {lda_params['random_state']}")
        print(f"  置信椭圆: {lda_params['ellipse_confidence']}")
        print(f"  按目标着色: {lda_params['color_by_target']}")
        print(f"  分析分离度: {lda_params['analyze_separation']}")
        print(f"  可视化判别向量: {lda_params['visualize_discriminants']}")
        print(f"  计算Fisher得分: {lda_params['calculate_fisher_score']}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("降维模型", "LDA").copy()

        # 如果用户提供了自定义图片参数，则更新
        if "image_param" in params and isinstance(params["image_param"], dict):
            plot_config.update(params["image_param"])

        # 3. 数据准备
        # 确定目标列
        y_column = lda_params["target_column"]

        # 检查目标列是否存在
        if y_column not in df.columns:
            error_msg = f"错误: LDA是监督降维方法，需要目标列 '{y_column}'。请检查数据框列名: {list(df.columns)}"
            raise ValueError(error_msg)

        # 确定特征列
        X_columns = [col for col in df.columns if col != y_column]

        # 检查特征列数量
        if len(X_columns) < 1:
            error_msg = f"错误: 特征列数量不足。目标列 '{y_column}' 之外的特征列只有 {len(X_columns)} 个，至少需要 1 个特征列进行LDA分析。"
            raise ValueError(error_msg)

        X = df[X_columns]
        y = df[y_column]

        print(f"\n使用目标列 '{y_column}' 进行监督降维")
        print(f"  特征数量: {len(X_columns)}")
        print(f"  目标变量类型: {y.dtype}")
        print(f"  目标变量类别数: {len(y.unique())}")

        # 检查类别数量
        n_classes = len(y.unique())
        if n_classes < 2:
            error_msg = f"错误: LDA需要至少2个类别，当前数据只有 {n_classes} 个类别。"
            raise ValueError(error_msg)

        # 检查特征数量与类别数的关系
        n_features = X.shape[1]
        max_components = min(n_features, n_classes - 1)

        print(f"\n数据信息:")
        print(f"  总样本数: {df.shape[0]}")
        print(f"  特征数据形状: {X.shape}")
        print(f"  目标数据形状: {y.shape}")
        print(f"  类别数量: {n_classes}")
        print(f"  LDA最大降维维度: {max_components}")

        # 对目标变量进行编码（如果是字符串类型）
        if y.dtype == 'object':
            print(f"目标变量为文本类型，正在进行标签编码...")
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            y_classes = label_encoder.classes_
            print(f"  编码后类别: {list(y_classes)}")
        else:
            y_encoded = y.values
            y_classes = np.unique(y_encoded)

        # 移除包含NaN的行
        if X.isna().any().any() or pd.isna(y_encoded).any():
            print(f"警告: 数据中包含NaN值，正在移除...")
            mask = ~(X.isna().any(axis=1) | pd.isna(y_encoded))
            X_clean = X[mask]
            y_clean = y_encoded[mask]
            print(f"  移除后样本数: {X_clean.shape[0]}")
            X = X_clean
            y_encoded = y_clean

        # 检查每个类别的样本数
        class_counts = np.bincount(y_encoded)
        print(f"\n类别分布:")
        for i, count in enumerate(class_counts):
            class_name = y_classes[i] if i < len(y_classes) else f"Class_{i}"
            print(f"  {class_name}: {count} 个样本 ({count / len(y_encoded):.1%})")

        # 检查是否有类别样本数过少
        min_samples_per_class = min(class_counts)
        if min_samples_per_class < 2:
            error_msg = f"错误: 某些类别样本数少于2个，LDA无法处理。"
            raise ValueError(error_msg)

        # 4. 数据标准化
        if lda_params["standardize"]:
            print("\n标准化特征数据...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # 5. 确定降维维度
        if lda_params["n_components"] is None:
            lda_params["n_components"] = max_components
            print(f"\n自动设置降维维度为: {max_components}")
        elif lda_params["n_components"] > max_components:
            print(
                f"\n警告: 指定的降维维度 {lda_params['n_components']} 超过最大值 {max_components}，将自动调整为 {max_components}")
            lda_params["n_components"] = max_components
        else:
            print(f"\n使用指定的降维维度: {lda_params['n_components']}")

        # 6. 应用LDA降维
        print(f"\n应用LDA降维...")

        # 准备LDA参数
        lda_kwargs = {
            "n_components": lda_params["n_components"],
            "solver": lda_params["solver"],
            "shrinkage": lda_params["shrinkage"],
            "priors": lda_params["priors"],
            "store_covariance": lda_params["store_covariance"],
            "tol": lda_params["tol"],
        }

        # 如果solver是'lsqr'或'eigen'，添加random_state
        if lda_params["solver"] in ["lsqr", "eigen"] and lda_params["random_state"] is not None:
            lda_kwargs["random_state"] = lda_params["random_state"]

        # 移除None值参数
        lda_kwargs = {k: v for k, v in lda_kwargs.items() if v is not None}

        lda_model = LDA(**lda_kwargs)
        X_lda = lda_model.fit_transform(X_scaled, y_encoded)

        # 获取解释方差比例
        if hasattr(lda_model, 'explained_variance_ratio_'):
            explained_variance_ratio = lda_model.explained_variance_ratio_
        else:
            # 手动计算解释方差比例
            # 对于LDA，解释方差比例可以通过判别式的特征值计算
            if hasattr(lda_model, 'scalings_'):
                # scalings_是判别向量
                explained_variance_ratio = None
            else:
                explained_variance_ratio = None

        print(f"\nLDA结果:")
        print(f"  目标维度: {lda_params['n_components']}")
        print(f"  求解器: {lda_params['solver']}")
        print(f"  类别数: {n_classes}")

        if explained_variance_ratio is not None:
            print(f"  总解释方差比例: {np.sum(explained_variance_ratio):.4f}")

        # 7. 生成二维投影散点图
        if lda_params["n_components"] >= 2:
            plt.figure(figsize=(12, 10))

            # 准备颜色映射
            if lda_params["color_by_target"]:
                unique_labels = np.unique(y_encoded)
                n_labels = len(unique_labels)
                colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
                color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
                point_colors = [color_map[label] for label in y_encoded]

                # 创建图例元素
                legend_elements = []
                for i, label in enumerate(unique_labels):
                    class_name = y_classes[label] if label < len(y_classes) else f"Class_{label}"
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                      markerfacecolor=colors[i],
                                                      markersize=10, label=class_name))
            else:
                point_colors = 'steelblue'
                legend_elements = []

            # 绘制散点图
            scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1],
                                  c=point_colors, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)

            # 添加置信椭圆
            if lda_params["color_by_target"] and lda_params["ellipse_confidence"] > 0:
                for label in unique_labels:
                    mask = (y_encoded == label)
                    if np.sum(mask) > 2:  # 至少需要3个点来绘制椭圆
                        x_vals = X_lda[mask, 0]
                        y_vals = X_lda[mask, 1]

                        # 使用卡方分布确定椭圆半径
                        n_std = np.sqrt(2) * np.sqrt(-2 * np.log(1 - lda_params["ellipse_confidence"]))

                        # 绘制置信椭圆
                        try:
                            confidence_ellipse(x_vals, y_vals, plt.gca(),
                                               n_std=n_std,
                                               edgecolor=color_map[label],
                                               linestyle='--',
                                               alpha=0.5)
                        except:
                            pass  # 如果无法绘制椭圆，跳过

            # 添加类别中心
            for label in unique_labels:
                mask = (y_encoded == label)
                center = np.mean(X_lda[mask, :2], axis=0)
                plt.scatter(center[0], center[1], s=200, marker='*',
                            color=color_map[label] if lda_params["color_by_target"] else 'red',
                            edgecolors='k', linewidth=1.5, zorder=10)

            # 添加标签和标题
            title = f"{plot_config['lda_2d_scatter_title']}\n"
            title += f"类别数: {n_classes}, 求解器: {lda_params['solver']}"

            plt.xlabel(plot_config["lda_2d_scatter_xlabel"], fontsize=12)
            plt.ylabel(plot_config["lda_2d_scatter_ylabel"], fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # 添加图例
            if legend_elements:
                plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

            # 添加解释文本
            text_str = f'样本数: {X_lda.shape[0]}\n特征数: {X.shape[1]}\n降维维度: {lda_params["n_components"]}'
            if explained_variance_ratio is not None and len(explained_variance_ratio) >= 2:
                text_str += f'\n前2个判别式解释方差: {np.sum(explained_variance_ratio[:2]):.1%}'
            plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            scatter_2d_path = os.path.join(output_dir, 'lda_2d_scatter.png')
            plt.savefig(scatter_2d_path, dpi=dpi)
            plt.close()

            # 保存二维投影数据
            scatter_2d_data_path = os.path.join(output_dir, 'lda_2d_projection_data.txt')
            scatter_2d_data = pd.DataFrame({
                'LD1': X_lda[:, 0],
                'LD2': X_lda[:, 1],
                'target': [y_classes[label] if label < len(y_classes) else f"Class_{label}" for label in y_encoded]
            })
            scatter_2d_data.to_csv(scatter_2d_data_path, sep='\t', index=False)

            imgs.append({
                "name": "LDA二维投影图",
                "img": scatter_2d_path,
                "data": scatter_2d_data_path
            })

        # 8. 生成三维投影散点图（如果降维维度>=3）
        if lda_params["n_components"] >= 3:
            try:
                from mpl_toolkits.mplot3d import Axes3D

                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')

                # 准备颜色映射
                if lda_params["color_by_target"]:
                    unique_labels = np.unique(y_encoded)
                    n_labels = len(unique_labels)
                    colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
                    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
                    point_colors = [color_map[label] for label in y_encoded]

                    # 创建图例元素（只显示前10个）
                    legend_elements = []
                    for i, label in enumerate(unique_labels[:10]):
                        class_name = y_classes[label] if label < len(y_classes) else f"Class_{label}"
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                          markerfacecolor=colors[i],
                                                          markersize=10, label=class_name))
                else:
                    point_colors = 'steelblue'
                    legend_elements = []

                # 绘制三维散点图
                scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], X_lda[:, 2],
                                     c=point_colors, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)

                # 添加类别中心
                if lda_params["color_by_target"]:
                    for label in unique_labels:
                        mask = (y_encoded == label)
                        center = np.mean(X_lda[mask, :3], axis=0)
                        ax.scatter(center[0], center[1], center[2], s=200, marker='*',
                                   color=color_map[label], edgecolors='k', linewidth=1.5, zorder=10)

                # 添加标签和标题
                title = f"{plot_config['lda_3d_scatter_title']}\n"
                title += f"类别数: {n_classes}, 求解器: {lda_params['solver']}"

                ax.set_xlabel(plot_config["lda_3d_scatter_xlabel"], fontsize=11)
                ax.set_ylabel(plot_config["lda_3d_scatter_ylabel"], fontsize=11)
                ax.set_zlabel(plot_config["lda_3d_scatter_zlabel"], fontsize=11)
                ax.set_title(title, fontsize=14, fontweight='bold')

                # 添加图例
                if legend_elements:
                    ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left')

                plt.tight_layout()
                scatter_3d_path = os.path.join(output_dir, 'lda_3d_scatter.png')
                plt.savefig(scatter_3d_path, dpi=dpi)
                plt.close()

                # 保存三维投影数据
                scatter_3d_data_path = os.path.join(output_dir, 'lda_3d_projection_data.txt')
                scatter_3d_data = pd.DataFrame({
                    'LD1': X_lda[:, 0],
                    'LD2': X_lda[:, 1],
                    'LD3': X_lda[:, 2],
                    'target': [y_classes[label] if label < len(y_classes) else f"Class_{label}" for label in y_encoded]
                })
                scatter_3d_data.to_csv(scatter_3d_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "LDA三维投影图",
                    "img": scatter_3d_path,
                    "data": scatter_3d_data_path
                })
            except Exception as e:
                print(f"生成三维散点图时出错: {e}")

        # 9. 分析LDA解释方差比例
        if hasattr(lda_model, 'explained_variance_ratio_') and explained_variance_ratio is not None:
            plt.figure(figsize=(12, 6))

            # 创建子图
            plt.subplot(1, 2, 1)

            # 解释方差比例条形图
            components_idx = range(1, len(explained_variance_ratio) + 1)
            bars = plt.bar(components_idx, explained_variance_ratio, alpha=0.7, color='steelblue')

            # 在柱子上添加数值标签
            for bar, val in zip(bars, explained_variance_ratio):
                height = bar.get_height()
                if val > 0.01:  # 只显示大于1%的值
                    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                             f'{val:.1%}', ha='center', va='bottom', fontsize=9)

            plt.xlabel(plot_config["explained_variance_xlabel"], fontsize=11)
            plt.ylabel(plot_config["explained_variance_ylabel"], fontsize=11)
            plt.title(plot_config["explained_variance_title"], fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')

            # 累计解释方差比例
            plt.subplot(1, 2, 2)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            plt.plot(components_idx, cumulative_variance, 'o-', linewidth=2, markersize=8, color='darkorange')
            plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.6, label='95%阈值')

            plt.xlabel(plot_config["explained_variance_xlabel"], fontsize=11)
            plt.ylabel("累计解释方差比例", fontsize=11)
            plt.title("累计解释方差比例", fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 标记达到95%的判别式数量
            if cumulative_variance[-1] >= 0.95:
                idx = np.where(cumulative_variance >= 0.95)[0][0]
                plt.axvline(x=components_idx[idx], color='g', linestyle=':', alpha=0.6)
                plt.text(components_idx[idx], 0.5, f'判别式{components_idx[idx]}\n达到95%',
                         ha='right', va='center', fontsize=9, color='g')

            plt.tight_layout()
            variance_plot_path = os.path.join(output_dir, 'lda_explained_variance.png')
            plt.savefig(variance_plot_path, dpi=dpi)
            plt.close()

            # 保存解释方差数据
            variance_data_path = os.path.join(output_dir, 'lda_explained_variance_data.txt')
            variance_data = pd.DataFrame({
                'component': components_idx,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance
            })
            variance_data.to_csv(variance_data_path, sep='\t', index=False)

            imgs.append({
                "name": "LDA解释方差分析",
                "img": variance_plot_path,
                "data": variance_data_path
            })

        # 10. 分析类别分离度
        if lda_params["analyze_separation"] and lda_params["n_components"] >= 2:
            try:
                # 计算类别间距离（马氏距离）
                n_classes = len(np.unique(y_encoded))
                class_means = []
                class_covs = []

                for label in np.unique(y_encoded):
                    mask = (y_encoded == label)
                    class_data = X_lda[mask, :2]  # 使用前两个判别式
                    class_means.append(np.mean(class_data, axis=0))
                    class_covs.append(np.cov(class_data.T))

                # 计算类别间马氏距离
                mahalanobis_distances = np.zeros((n_classes, n_classes))
                for i in range(n_classes):
                    for j in range(n_classes):
                        if i == j:
                            mahalanobis_distances[i, j] = 0
                        else:
                            try:
                                # 使用第一个类别的协方差矩阵的逆
                                inv_cov = np.linalg.pinv(class_covs[i])
                                diff = class_means[i] - class_means[j]
                                mahalanobis_distances[i, j] = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
                            except:
                                mahalanobis_distances[i, j] = np.nan

                plt.figure(figsize=(14, 6))

                # 子图1：马氏距离热图
                plt.subplot(1, 2, 1)
                sns.heatmap(mahalanobis_distances, annot=True, fmt='.2f', cmap='YlOrRd',
                            xticklabels=[f'C{i}' for i in range(n_classes)],
                            yticklabels=[f'C{i}' for i in range(n_classes)])
                plt.xlabel(plot_config["mahalanobis_distance_xlabel"], fontsize=11)
                plt.ylabel(plot_config["mahalanobis_distance_ylabel"], fontsize=11)
                plt.title(plot_config["mahalanobis_distance_title"], fontsize=12, fontweight='bold')

                # 子图2：类别中心可视化
                plt.subplot(1, 2, 2)

                # 绘制类别中心
                for i, mean in enumerate(class_means):
                    plt.scatter(mean[0], mean[1], s=200, marker='o',
                                color=plt.cm.tab20(i / max(1, n_classes - 1)),
                                edgecolors='k', linewidth=2, label=f'C{i}')

                # 连接类别中心
                for i in range(n_classes):
                    for j in range(i + 1, n_classes):
                        plt.plot([class_means[i][0], class_means[j][0]],
                                 [class_means[i][1], class_means[j][1]],
                                 'k--', alpha=0.3, linewidth=1)
                        # 在连线中点添加距离标签
                        mid_point = [(class_means[i][0] + class_means[j][0]) / 2,
                                     (class_means[i][1] + class_means[j][1]) / 2]
                        distance = mahalanobis_distances[i, j]
                        if not np.isnan(distance):
                            plt.text(mid_point[0], mid_point[1], f'{distance:.2f}',
                                     fontsize=9, ha='center', va='center',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                plt.xlabel(plot_config["class_centroids_xlabel"], fontsize=11)
                plt.ylabel(plot_config["class_centroids_ylabel"], fontsize=11)
                plt.title(plot_config["class_centroids_title"], fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.tight_layout()
                separation_plot_path = os.path.join(output_dir, 'lda_separation_analysis.png')
                plt.savefig(separation_plot_path, dpi=dpi)
                plt.close()

                # 保存分离度数据
                separation_data_path = os.path.join(output_dir, 'lda_separation_analysis_data.txt')
                separation_data = pd.DataFrame(mahalanobis_distances,
                                               columns=[f'Class_{i}' for i in range(n_classes)],
                                               index=[f'Class_{i}' for i in range(n_classes)])
                separation_data.to_csv(separation_data_path, sep='\t')

                imgs.append({
                    "name": "LDA类别分离度分析",
                    "img": separation_plot_path,
                    "data": separation_data_path
                })
            except Exception as e:
                print(f"分析类别分离度时出错: {e}")

        # 11. 可视化判别向量（如果可用）
        if lda_params["visualize_discriminants"] and hasattr(lda_model, 'scalings_'):
            try:
                scalings = lda_model.scalings_
                n_features = len(X_columns)
                n_components = scalings.shape[1]

                # 选择前10个最重要的特征（按第一个判别式的绝对值）
                if n_features > 10:
                    # 计算特征重要性（所有判别式的平均绝对权重）
                    feature_importance = np.mean(np.abs(scalings), axis=1)
                    top_indices = np.argsort(feature_importance)[-10:][::-1]
                    top_features = [X_columns[i] for i in top_indices]
                    top_scalings = scalings[top_indices, :min(3, n_components)]
                else:
                    top_indices = range(n_features)
                    top_features = X_columns
                    top_scalings = scalings[:, :min(3, n_components)]

                plt.figure(figsize=(12, 8))

                # 创建分组柱状图
                bar_width = 0.25
                index = np.arange(len(top_features))

                # 为前3个判别式创建柱状图
                colors = ['steelblue', 'darkorange', 'forestgreen']
                for i in range(min(3, n_components)):
                    offset = (i - 1) * bar_width
                    plt.bar(index + offset, top_scalings[:, i], bar_width,
                            color=colors[i], alpha=0.8, label=f'判别式 {i + 1}')

                plt.xlabel(plot_config["discriminant_vectors_xlabel"], fontsize=12)
                plt.ylabel(plot_config["discriminant_vectors_ylabel"], fontsize=12)
                plt.title(
                    f"{plot_config['discriminant_vectors_title']}\n前{len(top_features)}个特征对前{min(3, n_components)}个判别式的贡献",
                    fontsize=14, fontweight='bold')
                plt.xticks(index, top_features, rotation=45, ha='right', fontsize=10)
                plt.legend()
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()

                discriminant_plot_path = os.path.join(output_dir, 'lda_discriminant_vectors.png')
                plt.savefig(discriminant_plot_path, dpi=dpi)
                plt.close()

                # 保存判别向量数据
                discriminant_data_path = os.path.join(output_dir, 'lda_discriminant_vectors_data.txt')
                discriminant_data = pd.DataFrame(scalings,
                                                 columns=[f'LD{i + 1}' for i in range(n_components)],
                                                 index=X_columns)
                discriminant_data.to_csv(discriminant_data_path, sep='\t')

                imgs.append({
                    "name": "LDA判别向量分析",
                    "img": discriminant_plot_path,
                    "data": discriminant_data_path
                })
            except Exception as e:
                print(f"可视化判别向量时出错: {e}")

        # 12. 计算Fisher判别得分（特征选择指标）
        if lda_params["calculate_fisher_score"]:
            try:
                # Fisher得分：类间方差 / 类内方差
                n_classes = len(np.unique(y_encoded))
                n_features = X.shape[1]

                fisher_scores = np.zeros(n_features)
                overall_mean = np.mean(X_scaled, axis=0)

                for feature_idx in range(n_features):
                    feature_data = X_scaled[:, feature_idx]

                    # 计算类内方差（Within-class scatter）
                    within_class_var = 0
                    for label in np.unique(y_encoded):
                        mask = (y_encoded == label)
                        class_data = feature_data[mask]
                        within_class_var += np.var(class_data) * len(class_data)

                    # 计算类间方差（Between-class scatter）
                    between_class_var = 0
                    for label in np.unique(y_encoded):
                        mask = (y_encoded == label)
                        class_mean = np.mean(feature_data[mask])
                        class_size = np.sum(mask)
                        between_class_var += class_size * (class_mean - overall_mean[feature_idx]) ** 2

                    # 计算Fisher得分
                    if within_class_var > 0:
                        fisher_scores[feature_idx] = between_class_var / within_class_var
                    else:
                        fisher_scores[feature_idx] = 0

                # 可视化Fisher得分
                plt.figure(figsize=(12, 8))

                # 选择前20个最高得分的特征
                n_top = min(20, n_features)
                top_indices = np.argsort(fisher_scores)[-n_top:][::-1]
                top_scores = fisher_scores[top_indices]
                top_features = [X_columns[i] for i in top_indices]

                colors = ['steelblue' if score > np.mean(fisher_scores) else 'lightgray'
                          for score in top_scores]

                plt.barh(range(n_top), top_scores[::-1], color=colors[::-1])
                plt.yticks(range(n_top), top_features[::-1], fontsize=10)
                plt.xlabel(plot_config["fisher_score_ylabel"], fontsize=12)
                plt.ylabel(plot_config["fisher_score_xlabel"], fontsize=12)
                plt.title(f"{plot_config['fisher_score_title']}\n前{n_top}个最具判别力的特征",
                          fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='x')

                # 添加平均线
                plt.axvline(x=np.mean(fisher_scores), color='r', linestyle='--',
                            alpha=0.7, label=f'平均得分: {np.mean(fisher_scores):.3f}')
                plt.legend()

                plt.tight_layout()
                fisher_plot_path = os.path.join(output_dir, 'lda_fisher_scores.png')
                plt.savefig(fisher_plot_path, dpi=dpi)
                plt.close()

                # 保存Fisher得分数据
                fisher_data_path = os.path.join(output_dir, 'lda_fisher_scores_data.txt')
                fisher_data = pd.DataFrame({
                    'feature': X_columns,
                    'fisher_score': fisher_scores
                }).sort_values('fisher_score', ascending=False)
                fisher_data.to_csv(fisher_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "Fisher判别得分",
                    "img": fisher_plot_path,
                    "data": fisher_data_path
                })
            except Exception as e:
                print(f"计算Fisher得分时出错: {e}")

        # 13. 降维前后对比
        if X_scaled.shape[1] >= 2 and lda_params["n_components"] >= 2:
            try:
                # 选择两个与目标相关性最高的特征
                correlations = []
                for i in range(min(10, X_scaled.shape[1])):
                    if len(np.unique(y_encoded)) > 1:
                        # 使用ANOVA F值作为相关性度量
                        f_val, _ = f_oneway(*[X_scaled[y_encoded == label, i] for label in np.unique(y_encoded)])
                        correlations.append((i, f_val))
                    else:
                        correlations.append((i, 0))

                correlations.sort(key=lambda x: x[1], reverse=True)
                top_feature_indices = [correlations[0][0], correlations[1][0]] if len(correlations) >= 2 else [0, 1]

                plt.figure(figsize=(14, 6))

                # 子图1：原始特征空间
                plt.subplot(1, 2, 1)

                if lda_params["color_by_target"]:
                    unique_labels = np.unique(y_encoded)
                    n_labels = len(unique_labels)
                    colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
                    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
                    point_colors = [color_map[label] for label in y_encoded]
                else:
                    point_colors = 'steelblue'

                plt.scatter(X_scaled[:, top_feature_indices[0]], X_scaled[:, top_feature_indices[1]],
                            c=point_colors, alpha=0.7, s=30)
                plt.xlabel(f"{X_columns[top_feature_indices[0]]} (原始特征)", fontsize=11)
                plt.ylabel(f"{X_columns[top_feature_indices[1]]} (原始特征)", fontsize=11)
                plt.title("原始特征空间", fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)

                # 子图2：LDA投影空间
                plt.subplot(1, 2, 2)
                plt.scatter(X_lda[:, 0], X_lda[:, 1],
                            c=point_colors, alpha=0.7, s=30)
                plt.xlabel("线性判别式 1", fontsize=11)
                plt.ylabel("线性判别式 2", fontsize=11)
                plt.title("LDA投影空间", fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)

                plt.suptitle(plot_config["before_after_comparison_title"], fontsize=14, fontweight='bold')
                plt.tight_layout()

                comparison_path = os.path.join(output_dir, 'lda_before_after_comparison.png')
                plt.savefig(comparison_path, dpi=dpi)
                plt.close()

                imgs.append({
                    "name": "LDA降维前后对比",
                    "img": comparison_path,
                    "data": scatter_2d_data_path  # 使用二维投影数据
                })
            except Exception as e:
                print(f"生成降维前后对比图时出错: {e}")

        # 14. 保存LDA结果数据
        # 保存降维后的投影数据
        lda_projection_path = os.path.join(output_dir, 'lda_projection.txt')
        lda_projection_df = pd.DataFrame(X_lda, columns=[f'LD{i + 1}' for i in range(lda_params["n_components"])])
        if 'label_encoder' in locals():
            lda_projection_df['target'] = label_encoder.inverse_transform(y_encoded)
        else:
            lda_projection_df['target'] = y_encoded
        lda_projection_df.to_csv(lda_projection_path, sep='\t', index=False)

        # 15. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": X.shape[0],
                "features": X.shape[1],
                "target_variable": y_column,
                "feature_variables": X_columns,
                "target_classes": n_classes,
                "class_names": y_classes.tolist() if 'y_classes' in locals() else list(range(n_classes))
            },
            "model_params": lda_params,
            "lda_results": {
                "n_components": int(lda_params["n_components"]),
                "solver": lda_params["solver"],
                "explained_variance_ratio": explained_variance_ratio.tolist() if explained_variance_ratio is not None else None,
                "total_explained_variance": float(
                    np.sum(explained_variance_ratio)) if explained_variance_ratio is not None else None,
                "max_possible_components": int(max_components)
            },
            "files": {
                "lda_projection": os.path.basename(lda_projection_path),
                "lda_2d_data": os.path.basename(scatter_2d_data_path) if 'scatter_2d_data_path' in locals() else None,
                "fisher_scores": os.path.basename(fisher_data_path) if 'fisher_data_path' in locals() else None
            }
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 16. 生成结果字符串
        result_str += "=======================\n"
        result_str += "LDA模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征\n"
        result_str += f"2. 目标变量: {y_column} ({n_classes} 个类别)\n"
        result_str += f"3. 降维维度: {lda_params['n_components']}\n"
        result_str += f"4. 求解器: {lda_params['solver']}\n"

        if explained_variance_ratio is not None:
            result_str += f"5. 总解释方差比例: {np.sum(explained_variance_ratio):.4f}\n"
            if len(explained_variance_ratio) >= 2:
                result_str += f"6. 前2个判别式解释方差: {np.sum(explained_variance_ratio[:2]):.1%}\n"

        # 添加LDA算法特点
        result_str += f"\n=== LDA算法特点 ===\n"
        result_str += f"• 监督降维: 利用类别信息进行降维\n"
        result_str += f"• 最大化类间距离: 投影方向使得类间距离最大，类内距离最小\n"
        result_str += f"• 维度限制: 最多可降至 min(n_features, n_classes-1) 维\n"

        # 添加类别分离度信息
        result_str += f"\n=== 类别信息 ===\n"
        for i, count in enumerate(class_counts):
            class_name = y_classes[i] if i < len(y_classes) else f"Class_{i}"
            result_str += f"  {class_name}: {count} 个样本 ({count / len(y_encoded):.1%})\n"

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
        result_str += f"LDA投影结果文件: {os.path.basename(lda_projection_path)}\n"

        # 添加计算使用的参数信息
        result_str += f"\n=== 计算使用的参数信息 ===\n"
        result_str += f"LDA算法参数:\n"

        param_descriptions = {
            "target_column": f"目标列名 (当前: {lda_params['target_column']})",
            "features_only": f"是否只使用特征列 (当前: {lda_params['features_only']})",
            "standardize": f"是否标准化特征 (当前: {lda_params['standardize']})",
            "n_components": f"目标维度 (当前: {lda_params['n_components']})",
            "solver": f"求解器 (当前: {lda_params['solver']})",
            "shrinkage": f"收缩参数 (当前: {lda_params['shrinkage']})",
            "priors": f"先验概率 (当前: {lda_params['priors']})",
            "store_covariance": f"是否存储协方差 (当前: {lda_params['store_covariance']})",
            "tol": f"求解器容忍度 (当前: {lda_params['tol']})",
            "random_state": f"随机种子 (当前: {lda_params['random_state']})",
            "ellipse_confidence": f"置信椭圆置信水平 (当前: {lda_params['ellipse_confidence']})",
            "color_by_target": f"是否按目标着色 (当前: {lda_params['color_by_target']})",
            "analyze_separation": f"是否分析分离度 (当前: {lda_params['analyze_separation']})",
            "visualize_discriminants": f"是否可视化判别向量 (当前: {lda_params['visualize_discriminants']})",
            "calculate_fisher_score": f"是否计算Fisher得分 (当前: {lda_params['calculate_fisher_score']})"
        }

        for key, value in lda_params.items():
            if key in param_descriptions:
                result_str += f"  {key}: {value}\n"
                result_str += f"    说明: {param_descriptions[key]}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"LDA模型运行失败: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        result_str = "=======================\n"
        result_str += "LDA模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 降维模型 LDA 运行结束 <<<<<<<<<<<<<<<<<<<<")
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
            "lda_projection": {
                "data": lda_projection_path if 'lda_projection_path' in locals() else ""
            }
        },
        "start": start_timestamp,
        "end": end_timestamp,
        "expend": expend_time
    }

    # 将结果转换为JSON字符串返回
    return json.dumps(result_json, ensure_ascii=False, indent=2)