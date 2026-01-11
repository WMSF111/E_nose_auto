import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import warnings
from datetime import datetime
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh

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
    print(">>>>>>>>>>>>>>>>>>>> 降维模型 LLE 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"lle_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """LLE参数配置"""
        # 1. 扩展算法参数
        lle_params = {
            "target_column": "target",
            # 目标列名（用于着色）
            # 默认值：'target'
            # 注意：如果数据没有目标列，可以设置为None，则所有点使用相同颜色

            "features_only": False,
            # 是否只使用特征列（排除目标列）
            # 取值范围：布尔值 True 或 False
            #   True: 仅使用特征列进行LLE分析
            #   False: 包含所有列，但目标列仅用于着色
            # 默认值：False

            "standardize": True,
            # 是否标准化特征
            # 取值范围：布尔值 True 或 False
            #   True: 推荐，因为LLE对数据尺度敏感
            #   False: 使用原始数据尺度
            # 默认值：True

            "n_components": 2,
            # 目标维度（降维后的维度）
            # 取值范围：
            #   整数 (如 2, 3): 降维到指定维度
            #   通常用于可视化选择2或3维
            # 默认值：2

            "n_neighbors": 10,
            # 邻居数（局部邻域大小）
            # 取值范围：正整数，通常5-20
            #   - 过小：可能无法捕捉局部结构，导致不稳定的嵌入
            #   - 过大：可能破坏局部线性假设，失去非线性特性
            # 默认值：10

            "reg": 0.001,
            # 正则化系数
            # 取值范围：大于0的浮点数，通常1e-3到1e-2
            #   防止权重矩阵求解时的奇异性
            # 默认值：0.001

            "eigen_solver": "auto",
            # 特征值求解器
            # 可选值：
            #   'auto': 自动选择（默认）
            #   'arpack': 使用ARPACK求解器，适用于大规模数据
            #   'dense': 使用密集矩阵求解器，适用于小规模数据
            # 默认值：'auto'

            "tol": 1e-6,
            # 特征值求解器的容忍度
            # 取值范围：大于0的浮点数
            # 默认值：1e-6

            "max_iter": 100,
            # 最大迭代次数（仅当eigen_solver='arpack'时有效）
            # 取值范围：正整数
            # 默认值：100

            "method": "standard",
            # LLE方法变体
            # 可选值：
            #   'standard': 标准LLE（默认）
            #   'modified': 修改的LLE，增加稳定性
            #   'hessian': Hessian LLE，关注局部曲率
            #   'ltsa': 局部切空间对齐，保持全局几何结构
            # 默认值：'standard'

            "neighbors_algorithm": "auto",
            # 最近邻算法
            # 可选值：
            #   'auto': 自动选择（默认）
            #   'brute': 暴力搜索
            #   'kd_tree': KD树算法
            #   'ball_tree': 球树算法
            # 默认值：'auto'

            "random_state": None,
            # 随机种子
            # 取值范围：正整数或None
            #   None: 非确定性结果
            #   整数: 确保结果可重现
            # 默认值：None

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

            "analyze_neighbors": True,
            # 是否分析邻居数影响
            # 取值范围：布尔值 True 或 False
            #   True: 分析不同邻居数对结果的影响
            #   False: 仅使用指定邻居数
            # 默认值：True

            "neighbors_range": "5,15,5",
            # 邻居数分析范围
            # 格式："起始,结束,步长"
            # 示例："5,20,5" 表示分析5,10,15,20四个邻居数
            # 默认值："5,15,5"
        }

        # 从传入参数中更新LLE参数
        in_params = params.get("params", {})

        # 更新参数 - 处理可能的列表值
        if "target_column" in in_params:
            lle_params["target_column"] = in_params["target_column"]

        if "features_only" in in_params:
            lle_params["features_only"] = in_params["features_only"]

        if "standardize" in in_params:
            lle_params["standardize"] = in_params["standardize"]

        # n_components参数处理
        if "n_components" in in_params:
            n_comp_val = in_params["n_components"]
            lle_params["n_components"] = AlgModelParameters.format_to_int(n_comp_val)

        # n_neighbors参数处理
        if "n_neighbors" in in_params:
            n_neighbors_val = in_params["n_neighbors"]
            lle_params["n_neighbors"] = AlgModelParameters.format_to_int(n_neighbors_val)

        # reg参数处理
        if "reg" in in_params:
            reg_val = in_params["reg"]
            lle_params["reg"] = AlgModelParameters.format_to_float(reg_val)

        # eigen_solver参数处理
        if "eigen_solver" in in_params:
            eigen_val = in_params["eigen_solver"]
            if eigen_val in ["auto", "arpack", "dense"]:
                lle_params["eigen_solver"] = eigen_val

        # tol参数处理
        if "tol" in in_params:
            tol_val = in_params["tol"]
            lle_params["tol"] = AlgModelParameters.format_to_float(tol_val)

        # max_iter参数处理
        if "max_iter" in in_params:
            max_iter_val = in_params["max_iter"]
            lle_params["max_iter"] = AlgModelParameters.format_to_int(max_iter_val)

        # method参数处理
        if "method" in in_params:
            method_val = in_params["method"]
            if method_val in ["standard", "modified", "hessian", "ltsa"]:
                lle_params["method"] = method_val

        # neighbors_algorithm参数处理
        if "neighbors_algorithm" in in_params:
            neighbors_algo_val = in_params["neighbors_algorithm"]
            if neighbors_algo_val in ["auto", "brute", "kd_tree", "ball_tree"]:
                lle_params["neighbors_algorithm"] = neighbors_algo_val

        # random_state参数处理
        if "random_state" in in_params:
            random_val = in_params["random_state"]
            lle_params["random_state"] = AlgModelParameters.format_to_int(random_val)

        # ellipse_confidence参数处理
        if "ellipse_confidence" in in_params:
            ellipse_val = in_params["ellipse_confidence"]
            lle_params["ellipse_confidence"] = AlgModelParameters.format_to_float(ellipse_val)

        # color_by_target参数处理
        if "color_by_target" in in_params:
            color_val = in_params["color_by_target"]
            if isinstance(color_val, str):
                lle_params["color_by_target"] = (color_val.lower() == "true")
            else:
                lle_params["color_by_target"] = bool(color_val)

        # analyze_neighbors参数处理
        if "analyze_neighbors" in in_params:
            analyze_val = in_params["analyze_neighbors"]
            if isinstance(analyze_val, str):
                lle_params["analyze_neighbors"] = (analyze_val.lower() == "true")
            else:
                lle_params["analyze_neighbors"] = bool(analyze_val)

        # neighbors_range参数处理
        if "neighbors_range" in in_params:
            range_val = in_params["neighbors_range"]
            lle_params["neighbors_range"] = range_val

        # 打印参数信息
        print(f"\nLLE算法参数配置:")
        print(f"  目标列: {lle_params['target_column']}")
        print(f"  仅特征列: {lle_params['features_only']}")
        print(f"  标准化: {lle_params['standardize']}")
        print(f"  目标维度: {lle_params['n_components']}")
        print(f"  邻居数: {lle_params['n_neighbors']}")
        print(f"  正则化系数: {lle_params['reg']}")
        print(f"  LLE方法: {lle_params['method']}")
        print(f"  特征值求解器: {lle_params['eigen_solver']}")
        print(f"  最近邻算法: {lle_params['neighbors_algorithm']}")
        print(f"  随机种子: {lle_params['random_state']}")
        print(f"  置信椭圆: {lle_params['ellipse_confidence']}")
        print(f"  按目标着色: {lle_params['color_by_target']}")
        print(f"  分析邻居数影响: {lle_params['analyze_neighbors']}")
        if lle_params['analyze_neighbors']:
            print(f"  邻居数分析范围: {lle_params['neighbors_range']}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("降维模型", "LLE").copy()

        # 如果用户提供了自定义图片参数，则更新
        if "image_param" in params and isinstance(params["image_param"], dict):
            plot_config.update(params["image_param"])

        # 3. 数据准备
        # 确定目标列
        y_column = lle_params["target_column"]

        # 检查目标列是否存在
        y_exists = y_column in df.columns

        if lle_params["features_only"] or not y_exists:
            # 仅使用特征列或目标列不存在
            X_columns = [col for col in df.columns if col != "target"]
            X = df[X_columns]
            y = None
            print(f"\n仅使用特征列进行LLE分析")
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
                    y = pd.cut(y, bins=min(10, len(y.unique())), labels=False)
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
        if lle_params["standardize"]:
            print("\n标准化特征数据...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # 5. 应用LLE
        print(f"\n应用LLE降维...")

        # 准备LLE参数
        lle_kwargs = {
            "n_components": lle_params["n_components"],
            "n_neighbors": lle_params["n_neighbors"],
            "reg": lle_params["reg"],
            "eigen_solver": lle_params["eigen_solver"],
            "tol": lle_params["tol"],
            "max_iter": lle_params["max_iter"],
            "method": lle_params["method"],
            "neighbors_algorithm": lle_params["neighbors_algorithm"],
            "random_state": lle_params["random_state"]
        }

        # 移除None值参数
        lle_kwargs = {k: v for k, v in lle_kwargs.items() if v is not None}

        # 如果使用arpack求解器，添加异常处理
        if lle_kwargs.get("eigen_solver") == "arpack" or (
                lle_kwargs.get("eigen_solver") == "auto" and X_scaled.shape[0] > 1000):
            try:
                lle_model = LocallyLinearEmbedding(**lle_kwargs)
                X_lle = lle_model.fit_transform(X_scaled)
            except Exception as e:
                print(f"ARPACK求解器失败: {e}")
                print("尝试使用dense求解器...")
                lle_kwargs["eigen_solver"] = "dense"
                lle_model = LocallyLinearEmbedding(**lle_kwargs)
                X_lle = lle_model.fit_transform(X_scaled)
        else:
            lle_model = LocallyLinearEmbedding(**lle_kwargs)
            X_lle = lle_model.fit_transform(X_scaled)

        # 获取重构误差
        reconstruction_error = getattr(lle_model, 'reconstruction_error_', None)

        print(f"\nLLE结果:")
        print(f"  目标维度: {lle_params['n_components']}")
        print(f"  邻居数: {lle_params['n_neighbors']}")
        print(f"  LLE方法: {lle_params['method']}")
        if reconstruction_error is not None:
            print(f"  重构误差: {reconstruction_error:.6f}")

        # 6. 生成二维嵌入散点图
        if lle_params["n_components"] >= 2:
            plt.figure(figsize=(12, 10))

            # 准备颜色映射
            if y is not None and lle_params["color_by_target"]:
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
            scatter = plt.scatter(X_lle[:, 0], X_lle[:, 1],
                                  c=point_colors, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)

            # 添加置信椭圆
            if y is not None and lle_params["color_by_target"] and lle_params["ellipse_confidence"] > 0:
                for label in unique_labels:
                    mask = (y == label)
                    if np.sum(mask) > 2:  # 至少需要3个点来绘制椭圆
                        x_vals = X_lle[mask, 0]
                        y_vals = X_lle[mask, 1]

                        # 使用卡方分布确定椭圆半径
                        n_std = np.sqrt(2) * np.sqrt(-2 * np.log(1 - lle_params["ellipse_confidence"]))

                        # 绘制置信椭圆
                        confidence_ellipse(x_vals, y_vals, plt.gca(),
                                           n_std=n_std,
                                           edgecolor=color_map[label],
                                           linestyle='--',
                                           alpha=0.5)

            # 添加标签和标题
            title = f"{plot_config['lle_2d_scatter_title']}\n"
            title += f"邻居数: {lle_params['n_neighbors']}, 方法: {lle_params['method']}"
            if reconstruction_error is not None:
                title += f"\n重构误差: {reconstruction_error:.4f}"

            plt.xlabel(plot_config["lle_2d_scatter_xlabel"], fontsize=12)
            plt.ylabel(plot_config["lle_2d_scatter_ylabel"], fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # 添加图例
            if legend_elements:
                plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

            # 添加解释文本
            text_str = f'样本数: {X_lle.shape[0]}\n特征数: {X.shape[1]}\n目标维度: {lle_params["n_components"]}'
            plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            scatter_2d_path = os.path.join(output_dir, 'lle_2d_scatter.png')
            plt.savefig(scatter_2d_path, dpi=dpi)
            plt.close()

            # 保存二维嵌入数据
            scatter_2d_data_path = os.path.join(output_dir, 'lle_2d_embedding_data.txt')
            scatter_2d_data = pd.DataFrame({
                'LLE1': X_lle[:, 0],
                'LLE2': X_lle[:, 1],
                'target': y if y is not None else ['N/A'] * len(X_lle)
            })
            scatter_2d_data.to_csv(scatter_2d_data_path, sep='\t', index=False)

            imgs.append({
                "name": "LLE二维嵌入图",
                "img": scatter_2d_path,
                "data": scatter_2d_data_path
            })

        # 7. 生成三维嵌入散点图（如果目标维度>=3）
        if lle_params["n_components"] >= 3:
            try:
                from mpl_toolkits.mplot3d import Axes3D

                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')

                # 准备颜色映射
                if y is not None and lle_params["color_by_target"]:
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
                scatter = ax.scatter(X_lle[:, 0], X_lle[:, 1], X_lle[:, 2],
                                     c=point_colors, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)

                # 添加标签和标题
                title = f"{plot_config['lle_3d_scatter_title']}\n"
                title += f"邻居数: {lle_params['n_neighbors']}, 方法: {lle_params['method']}"
                if reconstruction_error is not None:
                    title += f"\n重构误差: {reconstruction_error:.4f}"

                ax.set_xlabel(plot_config["lle_3d_scatter_xlabel"], fontsize=11)
                ax.set_ylabel(plot_config["lle_3d_scatter_ylabel"], fontsize=11)
                ax.set_zlabel(plot_config["lle_3d_scatter_zlabel"], fontsize=11)
                ax.set_title(title, fontsize=14, fontweight='bold')

                # 添加图例
                if legend_elements:
                    ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left')

                plt.tight_layout()
                scatter_3d_path = os.path.join(output_dir, 'lle_3d_scatter.png')
                plt.savefig(scatter_3d_path, dpi=dpi)
                plt.close()

                # 保存三维嵌入数据
                scatter_3d_data_path = os.path.join(output_dir, 'lle_3d_embedding_data.txt')
                scatter_3d_data = pd.DataFrame({
                    'LLE1': X_lle[:, 0],
                    'LLE2': X_lle[:, 1],
                    'LLE3': X_lle[:, 2],
                    'target': y if y is not None else ['N/A'] * len(X_lle)
                })
                scatter_3d_data.to_csv(scatter_3d_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "LLE三维嵌入图",
                    "img": scatter_3d_path,
                    "data": scatter_3d_data_path
                })
            except Exception as e:
                print(f"生成三维散点图时出错: {e}")

        # 8. 分析邻居数影响
        if lle_params["analyze_neighbors"] and lle_params["n_components"] >= 2:
            try:
                # 解析邻居数范围
                range_parts = lle_params["neighbors_range"].split(',')
                if len(range_parts) == 3:
                    start = int(range_parts[0])
                    end = int(range_parts[1])
                    step = int(range_parts[2])
                else:
                    # 默认范围
                    start = 5
                    end = 15
                    step = 5

                neighbors_values = list(range(start, end + 1, step))
                if lle_params["n_neighbors"] not in neighbors_values:
                    neighbors_values.append(lle_params["n_neighbors"])
                neighbors_values = sorted(set(neighbors_values))

                print(f"\n分析邻居数影响: {neighbors_values}")

                # 存储结果
                reconstruction_errors = []
                embeddings = {}

                for n_neighbors in neighbors_values:
                    print(f"  计算邻居数 {n_neighbors}...")

                    try:
                        # 尝试使用dense求解器，它更稳定但计算量更大
                        lle_temp = LocallyLinearEmbedding(
                            n_components=min(3, lle_params["n_components"]),
                            n_neighbors=n_neighbors,
                            reg=lle_params["reg"],
                            method=lle_params["method"],
                            eigen_solver='dense',  # 使用dense求解器避免奇异矩阵问题
                            random_state=lle_params["random_state"]
                        )

                        # 使用数据子集，并确保至少有两个特征
                        max_features = min(50, X_scaled.shape[1])
                        if max_features < 2:  # 确保至少有两个特征
                            max_features = min(2, X_scaled.shape[1])

                        # 应用LLE
                        X_lle_temp = lle_temp.fit_transform(X_scaled[:, :max_features])

                        # 存储结果
                        reconstruction_error_temp = getattr(lle_temp, 'reconstruction_error_', None)
                        reconstruction_errors.append(reconstruction_error_temp)
                        embeddings[n_neighbors] = X_lle_temp

                    except Exception as e:
                        print(f"    邻居数 {n_neighbors} 计算失败: {e}")
                        # 尝试使用更稳定的配置
                        try:
                            # 增加正则化系数并减小邻居数
                            lle_temp = LocallyLinearEmbedding(
                                n_components=min(2, lle_params["n_components"]),  # 减少维度
                                n_neighbors=min(n_neighbors, 5),  # 减小邻居数
                                reg=0.1,  # 增加正则化系数
                                method=lle_params["method"],
                                eigen_solver='dense',
                                random_state=lle_params["random_state"]
                            )

                            max_features = min(20, X_scaled.shape[1])  # 使用更少的特征
                            if max_features < 2:
                                max_features = min(2, X_scaled.shape[1])

                            X_lle_temp = lle_temp.fit_transform(X_scaled[:min(200, len(X_scaled)), :max_features])
                            reconstruction_error_temp = getattr(lle_temp, 'reconstruction_error_', None)
                            reconstruction_errors.append(reconstruction_error_temp)
                            embeddings[n_neighbors] = X_lle_temp
                            print(f"    使用备用配置成功计算邻居数 {n_neighbors}")
                        except Exception as e2:
                            print(f"    备用配置也失败: {e2}")
                            reconstruction_errors.append(None)
                            embeddings[n_neighbors] = None

                # 绘制邻居数分析图
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))

                # 子图1：重构误差随邻居数变化
                ax1 = axes[0, 0]
                valid_indices = [i for i, err in enumerate(reconstruction_errors) if err is not None]
                if valid_indices:
                    valid_neighbors = [neighbors_values[i] for i in valid_indices]
                    valid_errors = [reconstruction_errors[i] for i in valid_indices]
                    ax1.plot(valid_neighbors, valid_errors, 'bo-', linewidth=2, markersize=8)
                    ax1.axvline(x=lle_params["n_neighbors"], color='r', linestyle='--',
                                alpha=0.7, label=f'当前邻居数: {lle_params["n_neighbors"]}')
                    ax1.set_xlabel(plot_config["neighbors_analysis_xlabel"], fontsize=11)
                    ax1.set_ylabel(plot_config["neighbors_analysis_ylabel"], fontsize=11)
                    ax1.set_title(plot_config["neighbors_analysis_title"], fontsize=12, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()

                # 子图2：重构误差分布
                ax2 = axes[0, 1]
                if reconstruction_error is not None:
                    # 计算每个样本的重构误差（近似）
                    n_neighbors = lle_params["n_neighbors"]
                    neigh = NearestNeighbors(n_neighbors=n_neighbors + 1)
                    neigh.fit(X_scaled)
                    distances, indices = neigh.kneighbors(X_scaled)

                    # 计算局部重建误差（近似）
                    local_errors = []
                    for i in range(X_scaled.shape[0]):
                        # 获取邻居索引（排除自身）
                        neighbor_indices = indices[i, 1:]
                        # 计算重建权重（简化版本）
                        weights = np.ones(n_neighbors) / n_neighbors
                        # 计算重建点
                        reconstructed = np.dot(weights, X_scaled[neighbor_indices])
                        # 计算误差
                        error = np.linalg.norm(X_scaled[i] - reconstructed)
                        local_errors.append(error)

                    ax2.hist(local_errors, bins=30, edgecolor='black', alpha=0.7)
                    ax2.axvline(x=np.mean(local_errors), color='r', linestyle='--',
                                label=f'均值: {np.mean(local_errors):.4f}')
                    ax2.set_xlabel(plot_config["reconstruction_error_xlabel"], fontsize=11)
                    ax2.set_ylabel("频率", fontsize=11)
                    ax2.set_title(plot_config["reconstruction_error_title"], fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()

                # 子图3：不同邻居数的嵌入结果对比（二维投影）
                ax3 = axes[1, 0]
                n_plots = min(3, len(neighbors_values))
                colors = plt.cm.tab10(np.linspace(0, 1, n_plots))

                for i, n_neighbors in enumerate(neighbors_values[:n_plots]):
                    if n_neighbors in embeddings and embeddings[n_neighbors] is not None:
                        embedding = embeddings[n_neighbors]
                        ax3.scatter(embedding[:, 0], embedding[:, 1],
                                    alpha=0.6, s=30, color=colors[i],
                                    label=f'k={n_neighbors}')

                ax3.set_xlabel("嵌入维度1", fontsize=11)
                ax3.set_ylabel("嵌入维度2", fontsize=11)
                ax3.set_title("不同邻居数的嵌入结果对比", fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend()

                # 子图4：最近邻图可视化
                ax4 = axes[1, 1]
                if X_scaled.shape[1] >= 2:
                    # 使用前两个特征进行可视化
                    ax4.scatter(X_scaled[:100, 0], X_scaled[:100, 1],
                                alpha=0.6, s=50, c='blue')

                    # 绘制最近邻连接（只显示部分点）
                    n_points = min(20, X_scaled.shape[0])
                    for i in range(n_points):
                        # 找到最近邻
                        distances = np.linalg.norm(X_scaled[:n_points] - X_scaled[i], axis=1)
                        nearest_indices = np.argsort(distances)[1:4]  # 3个最近邻

                        for j in nearest_indices:
                            ax4.plot([X_scaled[i, 0], X_scaled[j, 0]],
                                     [X_scaled[i, 1], X_scaled[j, 1]],
                                     'r-', alpha=0.3, linewidth=0.5)

                    ax4.set_xlabel(plot_config["neighborhood_graph_xlabel"], fontsize=11)
                    ax4.set_ylabel(plot_config["neighborhood_graph_ylabel"], fontsize=11)
                    ax4.set_title(plot_config["neighborhood_graph_title"], fontsize=12, fontweight='bold')
                    ax4.grid(True, alpha=0.3)

                plt.tight_layout()
                neighbors_analysis_path = os.path.join(output_dir, 'lle_neighbors_analysis.png')
                plt.savefig(neighbors_analysis_path, dpi=dpi)
                plt.close()

                # 保存邻居数分析数据
                neighbors_data_path = os.path.join(output_dir, 'lle_neighbors_analysis_data.txt')
                neighbors_data = pd.DataFrame({
                    'n_neighbors': neighbors_values,
                    'reconstruction_error': reconstruction_errors
                })
                neighbors_data.to_csv(neighbors_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "LLE邻居数分析",
                    "img": neighbors_analysis_path,
                    "data": neighbors_data_path
                })

            except Exception as e:
                print(f"分析邻居数影响时出错: {e}")

        # 9. 生成参数敏感性分析图
        try:
            # 分析正则化系数reg的影响
            reg_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            reg_errors = []

            for reg_val in reg_values:
                lle_temp = LocallyLinearEmbedding(
                    n_components=2,
                    n_neighbors=lle_params["n_neighbors"],
                    reg=reg_val,
                    method=lle_params["method"],
                    random_state=lle_params["random_state"]
                )

                # 使用子集加快计算
                subset_size = min(500, X_scaled.shape[0])
                X_subset = X_scaled[:subset_size]

                try:
                    X_lle_temp = lle_temp.fit_transform(X_subset)
                    reconstruction_error_temp = getattr(lle_temp, 'reconstruction_error_', None)
                    reg_errors.append(reconstruction_error_temp)
                except:
                    reg_errors.append(None)

            plt.figure(figsize=(10, 6))

            # 绘制正则化系数影响
            valid_indices = [i for i, err in enumerate(reg_errors) if err is not None]
            if valid_indices:
                valid_regs = [reg_values[i] for i in valid_indices]
                valid_errors = [reg_errors[i] for i in valid_indices]

                plt.semilogx(valid_regs, valid_errors, 'go-', linewidth=2, markersize=8)
                plt.axvline(x=lle_params["reg"], color='r', linestyle='--',
                            alpha=0.7, label=f'当前reg: {lle_params["reg"]}')
                plt.xlabel(plot_config["parameter_sensitivity_xlabel"], fontsize=12)
                plt.ylabel(plot_config["parameter_sensitivity_ylabel"], fontsize=12)
                plt.title(f"{plot_config['parameter_sensitivity_title']}\n正则化系数影响",
                          fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, which='both')
                plt.legend()

                param_sensitivity_path = os.path.join(output_dir, 'lle_parameter_sensitivity.png')
                plt.tight_layout()
                plt.savefig(param_sensitivity_path, dpi=dpi)
                plt.close()

                # 保存参数敏感性数据
                param_data_path = os.path.join(output_dir, 'lle_parameter_sensitivity_data.txt')
                param_data = pd.DataFrame({
                    'reg': valid_regs,
                    'reconstruction_error': valid_errors
                })
                param_data.to_csv(param_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "LLE参数敏感性分析",
                    "img": param_sensitivity_path,
                    "data": param_data_path
                })
        except Exception as e:
            print(f"生成参数敏感性分析图时出错: {e}")

        # 10. 保存LLE结果数据
        # 保存降维后的嵌入结果
        lle_scores_path = os.path.join(output_dir, 'lle_embedding.txt')
        lle_scores_df = pd.DataFrame(X_lle, columns=[f'LLE{i + 1}' for i in range(lle_params["n_components"])])
        if y is not None:
            lle_scores_df['target'] = y
        lle_scores_df.to_csv(lle_scores_path, sep='\t', index=False)

        # 11. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": X.shape[0],
                "features": X.shape[1],
                "target_variable": y_column if y_exists else None,
                "feature_variables": X_columns,
                "target_used": y is not None
            },
            "model_params": lle_params,
            "lle_results": {
                "n_components": int(lle_params["n_components"]),
                "n_neighbors": int(lle_params["n_neighbors"]),
                "method": lle_params["method"],
                "reconstruction_error": float(reconstruction_error) if reconstruction_error is not None else None,
                "embedding_dimensions": lle_params["n_components"]
            },
            "files": {
                "lle_embedding": os.path.basename(lle_scores_path),
                "lle_2d_data": os.path.basename(scatter_2d_data_path) if 'scatter_2d_data_path' in locals() else None,
                "neighbors_analysis_data": os.path.basename(
                    neighbors_data_path) if 'neighbors_data_path' in locals() else None
            }
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 12. 生成结果字符串
        result_str += "=======================\n"
        result_str += "LLE模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征\n"
        result_str += f"2. 目标变量: {y_column if y_exists else '无'}\n"
        result_str += f"3. 降维维度: {lle_params['n_components']}\n"
        result_str += f"4. 邻居数: {lle_params['n_neighbors']}\n"
        result_str += f"5. LLE方法: {lle_params['method']}\n"

        if reconstruction_error is not None:
            result_str += f"6. 重构误差: {reconstruction_error:.6f}\n"

        # 添加LLE算法特点
        result_str += f"\n=== LLE算法特点 ===\n"
        result_str += f"• 非线性降维: 能够处理复杂的流形结构\n"
        result_str += f"• 局部保持: 保持数据点的局部邻域关系\n"
        result_str += f"• 参数敏感: 邻居数对结果影响较大\n"

        # 添加邻居数选择建议
        result_str += f"\n=== 邻居数选择建议 ===\n"
        if lle_params["n_neighbors"] < 5:
            result_str += f"当前邻居数({lle_params['n_neighbors']})可能偏小，建议尝试5-15\n"
        elif lle_params["n_neighbors"] > 20:
            result_str += f"当前邻居数({lle_params['n_neighbors']})可能偏大，可能破坏局部线性假设\n"
        else:
            result_str += f"当前邻居数({lle_params['n_neighbors']})在合理范围内\n"

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
        result_str += f"LLE嵌入结果文件: {os.path.basename(lle_scores_path)}\n"

        # 添加计算使用的参数信息
        result_str += f"\n=== 计算使用的参数信息 ===\n"
        result_str += f"LLE算法参数:\n"

        param_descriptions = {
            "target_column": f"目标列名 (当前: {lle_params['target_column']})",
            "features_only": f"是否只使用特征列 (当前: {lle_params['features_only']})",
            "standardize": f"是否标准化特征 (当前: {lle_params['standardize']})",
            "n_components": f"目标维度 (当前: {lle_params['n_components']})",
            "n_neighbors": f"邻居数 (当前: {lle_params['n_neighbors']})",
            "reg": f"正则化系数 (当前: {lle_params['reg']})",
            "method": f"LLE方法 (当前: {lle_params['method']})",
            "eigen_solver": f"特征值求解器 (当前: {lle_params['eigen_solver']})",
            "neighbors_algorithm": f"最近邻算法 (当前: {lle_params['neighbors_algorithm']})",
            "random_state": f"随机种子 (当前: {lle_params['random_state']})",
            "ellipse_confidence": f"置信椭圆置信水平 (当前: {lle_params['ellipse_confidence']})",
            "color_by_target": f"是否按目标着色 (当前: {lle_params['color_by_target']})",
            "analyze_neighbors": f"是否分析邻居数影响 (当前: {lle_params['analyze_neighbors']})"
        }

        for key, value in lle_params.items():
            if key in param_descriptions:
                result_str += f"  {key}: {value}\n"
                result_str += f"    说明: {param_descriptions[key]}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"LLE模型运行失败: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        result_str = "=======================\n"
        result_str += "LLE模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 降维模型 LLE 运行结束 <<<<<<<<<<<<<<<<<<<<")
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
            "lle_embedding": {
                "data": lle_scores_path if 'lle_scores_path' in locals() else ""
            }
        },
        "start": start_timestamp,
        "end": end_timestamp,
        "expend": expend_time
    }

    # 将结果转换为JSON字符串返回
    return json.dumps(result_json, ensure_ascii=False, indent=2)