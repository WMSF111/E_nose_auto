import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy import stats
import joblib

warnings.filterwarnings('ignore')

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 预测模型 SVR 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"svr_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """SVR参数配置"""
        # 1. 扩展算法参数
        svr_params = {
            "target_column": "sensor16",
            # 目标列名
            # 默认值：'sensor16'

            "test_size": 0.2,
            # 测试集比例
            # 取值范围：0到1之间的浮点数，例如 0.1, 0.2, 0.3
            # 默认值：0.2

            "random_state": 42,
            # 随机种子
            # 取值范围：正整数或None，例如 42, 123, None
            # 默认值：42

            "cv_folds": 5,
            # 交叉验证折数
            # 取值范围：正整数，通常3-10，例如 3, 5, 10
            # 默认值：5

            "standardize": True,
            # 是否标准化特征
            # 取值范围：布尔值 True 或 False
            # 默认值：True

            "kernel": "rbf",
            # 核函数类型
            # 可选值：
            #   'linear'    : 线性核函数，适用于线性可分问题，计算速度快
            #   'poly'      : 多项式核函数，适用于非线性问题，可调参数度(degree)
            #   'rbf'       : 径向基函数核（默认），适用于大多数非线性问题，泛化能力强
            #   'sigmoid'   : sigmoid核函数，类似于神经网络激活函数
            # 默认值：'rbf'

            "C": 1.0,
            # 正则化参数（惩罚系数）
            # 取值范围：大于0的浮点数，通常 0.001 到 1000
            #   - 较小的C：允许更多错误，模型更简单，可能欠拟合
            #   - 较大的C：惩罚错误更严格，模型更复杂，可能过拟合
            # 常用值：0.01, 0.1, 1.0, 10, 100, 1000
            # 默认值：1.0

            "epsilon": 0.1,
            # epsilon-不敏感损失函数的epsilon值
            # 取值范围：大于0的浮点数，通常 0.01 到 0.5
            #   - 较小的epsilon：对误差更敏感，拟合更精确，但可能过拟合
            #   - 较大的epsilon：对误差更容忍，模型更简单，泛化能力更强
            # 常用值：0.01, 0.05, 0.1, 0.2, 0.5
            # 默认值：0.1

            "gamma": "scale",
            # 核函数系数
            # 可选值：
            #   'scale'     : 1 / (n_features * X.var())，推荐用于特征方差不同的情况
            #   'auto'      : 1 / n_features，旧版本默认值
            #   浮点数      : 自定义gamma值，例如 0.01, 0.1, 1, 10
            #   - 较小的gamma：决策边界更平滑，模型更简单
            #   - 较大的gamma：决策边界更复杂，模型更复杂
            # 默认值：'scale'

            "degree": 3,
            # 多项式核函数的度（仅当kernel='poly'时有效）
            # 取值范围：正整数，通常 2 到 5
            #   - 较低的度：模型更简单
            #   - 较高的度：模型更复杂，可能过拟合
            # 默认值：3

            "coef0": 0.0,
            # 核函数中的独立项（仅当kernel='poly'或'sigmoid'时有效）
            # 取值范围：浮点数，通常 -1.0 到 1.0
            # 默认值：0.0

            "grid_search": False,
            # 是否进行网格搜索调参
            # 取值范围：布尔值 True 或 False
            #   True: 自动搜索最佳超参数组合，但计算时间较长
            #   False: 使用手动设置的参数
            # 默认值：False

            "grid_search_cv": 3,
            # 网格搜索交叉验证折数（仅当grid_search=True时有效）
            # 取值范围：正整数，通常 3 到 5
            # 默认值：3

            "cache_size": 200,
            # 核缓存大小（MB）
            # 取值范围：正整数，通常 100 到 1000
            #   - 较大的缓存：加快训练速度，但占用更多内存
            # 默认值：200

            "max_iter": -1,
            # 最大迭代次数
            # 取值范围：正整数或-1
            #   -1: 无限制迭代，直到收敛
            #   正整数: 最大迭代次数，例如 1000, 10000
            # 默认值：-1

            "tol": 0.001,
            # 收敛容差
            # 取值范围：大于0的浮点数，通常 1e-4 到 1e-2
            #   - 较小的tol：更精确的收敛，但可能需要更多迭代
            #   - 较大的tol：更快收敛，但可能精度较低
            # 默认值：0.001

            "shrinking": True,
            # 是否使用收缩启发式
            # 取值范围：布尔值 True 或 False
            #   True: 使用收缩启发式，加快训练速度
            #   False: 不使用收缩启发式
            # 默认值：True
        }

        # 从传入参数中更新SVR参数
        in_params = params.get("params", {})

        # 更新参数 - 处理可能的列表值
        if "target_column" in in_params:
            svr_params["target_column"] = in_params["target_column"]

        if "test_size" in in_params:
            test_val = in_params["test_size"]
            svr_params["test_size"] = AlgModelParameters.format_to_float(test_val)

        if "random_state" in in_params:
            random_val = in_params["random_state"]
            svr_params["random_state"] = AlgModelParameters.format_to_int(random_val)

        if "cv_folds" in in_params:
            cv_val = in_params["cv_folds"]
            svr_params["cv_folds"] = AlgModelParameters.format_to_int(cv_val)

        if "standardize" in in_params:
            svr_params["standardize"] = in_params["standardize"]

        # SVR特定参数
        if "kernel" in in_params:
            svr_params["kernel"] = in_params.get("kernel", "rbf")

        if "C" in in_params:
            C_val = in_params.get("C", 1.0)
            svr_params["C"] = AlgModelParameters.format_to_float(C_val)

        if "epsilon" in in_params:
            epsilon_val = in_params["epsilon"]
            svr_params["epsilon"] = AlgModelParameters.format_to_float(epsilon_val)

        if "gamma" in in_params:
            gamma_val = in_params["gamma"]
            if gamma_val in ["scale", "auto"]:
                svr_params["gamma"] = gamma_val
            else:
                svr_params["gamma"] = AlgModelParameters.format_to_float(gamma_val)

        if "degree" in in_params:
            degree_val = in_params["degree"]
            svr_params["degree"] = AlgModelParameters.format_to_int(degree_val)

        if "coef0" in in_params:
            coef0_val = in_params["coef0"]
            svr_params["coef0"] = AlgModelParameters.format_to_float(coef0_val)

        if "grid_search" in in_params:
            svr_params["grid_search"] = in_params["grid_search"]

        if "grid_search_cv" in in_params:
            gs_cv_val = in_params["grid_search_cv"]
            svr_params["grid_search_cv"] = AlgModelParameters.format_to_int(gs_cv_val)

        if "cache_size" in in_params:
            cache_val = in_params["cache_size"]
            svr_params["cache_size"] = AlgModelParameters.format_to_int(cache_val)

        if "max_iter" in in_params:
            max_iter_val = in_params["max_iter"]
            svr_params["max_iter"] = AlgModelParameters.format_to_int(max_iter_val)

        if "tol" in in_params:
            tol_val = in_params["tol"]
            svr_params["tol"] = AlgModelParameters.format_to_float(tol_val)

        if "shrinking" in in_params:
            svr_params["shrinking"] = in_params["shrinking"]

        # 打印参数信息
        print(f"\nSVR算法参数配置:")
        print(f"  目标列: {svr_params['target_column']}")
        print(f"  测试集比例: {svr_params['test_size']}")
        print(f"  随机种子: {svr_params['random_state']}")
        print(f"  核函数: {svr_params['kernel']}")
        print(f"  正则化参数C: {svr_params['C']}")
        print(f"  epsilon: {svr_params['epsilon']}")
        print(f"  gamma: {svr_params['gamma']}")
        if svr_params['kernel'] == 'poly':
            print(f"  多项式度: {svr_params['degree']}")
            print(f"  系数项: {svr_params['coef0']}")
        print(f"  网格搜索: {svr_params['grid_search']}")
        if svr_params['grid_search']:
            print(f"  网格搜索CV折数: {svr_params['grid_search_cv']}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("预测模型", "SVR").copy()

        # 如果用户提供了自定义图片参数，则更新
        if "image_param" in params and isinstance(params["image_param"], dict):
            plot_config.update(params["image_param"])

        # 3. 数据准备
        # 确定目标列
        y_column = svr_params["target_column"]

        # 检查目标列是否存在
        if y_column not in df.columns:
            # 尝试使用'sensor16'作为默认
            if 'sensor16' in df.columns:
                y_column = 'sensor16'
                svr_params["target_column"] = 'sensor16'
                print(f"警告: 目标列 '{svr_params['target_column']}' 不存在，使用默认列 'sensor16'")
            else:
                error_msg = f"错误: 数据框中不存在目标列 '{svr_params['target_column']}' 或 'sensor16'。请检查数据框列名: {list(df.columns)}"
                raise ValueError(error_msg)

        # 确定特征列 - 除了目标列之外的所有列
        X_columns = [col for col in df.columns if col != y_column and col != "target"]

        # 检查特征列数量
        if len(X_columns) < 1:
            error_msg = f"错误: 特征列数量不足。目标列 '{y_column}' 之外的特征列只有 {len(X_columns)} 个，至少需要 1 个特征列进行建模。"
            raise ValueError(error_msg)

        X = df[X_columns]
        y = df[y_column]

        # 将目标变量转换为数值（如果需要）
        if y.dtype == 'object':
            print("检测到目标变量为文本类型，尝试转换为数值...")
            try:
                y = pd.to_numeric(y, errors='coerce')
                # 检查转换结果
                nan_count = y.isna().sum()
                if nan_count > 0:
                    print(f"警告: 有 {nan_count} 个目标值无法转换为数值，将被移除")
                    # 移除NaN值
                    valid_mask = ~y.isna()
                    X = X[valid_mask]
                    y = y[valid_mask]
            except Exception as e:
                raise ValueError(f"目标变量转换失败: {e}")

        print(f"\n数据信息:")
        print(f"  目标变量: {y_column}")
        print(f"  特征数量: {len(X_columns)}")
        print(f"  特征列: {X_columns}")
        print(f"  总样本数: {df.shape[0]}")
        print(f"  特征数据形状: {X.shape}")
        print(f"  目标数据形状: {y.shape}")
        print(f"  目标变量类型: {y.dtype}")
        print(f"  目标变量范围: {y.min():.4f} - {y.max():.4f}")

        # 移除包含NaN的行
        data_combined = pd.concat([X, y], axis=1)
        data_cleaned = data_combined.dropna()
        if len(data_cleaned) < len(data_combined):
            print(f"警告: 移除了 {len(data_combined) - len(data_cleaned)} 个包含NaN的行")
            X = data_cleaned[X_columns]
            y = data_cleaned[y_column]

        # 4. 数据标准化
        if svr_params["standardize"]:
            print("\n标准化特征数据...")
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            # 保存标准化器
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(scaler_X, scaler_path)
        else:
            X_scaled = X.values

        # 5. 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=svr_params["test_size"],
            random_state=svr_params["random_state"]
        )

        print(f"\n数据集分割:")
        print(f"  训练集大小: {X_train.shape[0]} ({1 - svr_params['test_size']:.0%})")
        print(f"  测试集大小: {X_test.shape[0]} ({svr_params['test_size']:.0%})")

        # 6. 训练SVR模型
        print(f"\n训练SVR模型...")

        # 准备基础模型参数
        base_params = {
            "kernel": svr_params["kernel"],
            "C": svr_params["C"],
            "epsilon": svr_params["epsilon"],
            "gamma": svr_params["gamma"],
            "degree": svr_params["degree"],
            "coef0": svr_params["coef0"],
            "cache_size": svr_params["cache_size"],
            "max_iter": svr_params["max_iter"],
            "tol": svr_params["tol"],
            "shrinking": svr_params["shrinking"]
        }

        # 网格搜索调参
        if svr_params["grid_search"]:
            print("进行网格搜索调参...")
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': [0.01, 0.1, 1.0, 'scale'],
                'epsilon': [0.01, 0.1, 0.2]
            }

            grid_search = GridSearchCV(
                SVR(kernel=svr_params["kernel"]),
                param_grid,
                cv=svr_params["grid_search_cv"],
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            svr_model = grid_search.best_estimator_

            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

            # 更新参数
            svr_params["C"] = grid_search.best_params_.get("C", svr_params["C"])
            svr_params["gamma"] = grid_search.best_params_.get("gamma", svr_params["gamma"])
            svr_params["epsilon"] = grid_search.best_params_.get("epsilon", svr_params["epsilon"])

            # 保存网格搜索结果
            cv_results_path = os.path.join(output_dir, 'grid_search_results.txt')
            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            cv_results_df.to_csv(cv_results_path, sep='\t', index=False)

            # 生成超参数热力图
            if 'param_C' in cv_results_df.columns and 'param_gamma' in cv_results_df.columns:
                # 提取C和gamma的独特值
                C_values = sorted([c for c in cv_results_df['param_C'].unique() if isinstance(c, (int, float))])
                gamma_values = [g for g in cv_results_df['param_gamma'].unique() if
                                g in ['scale', 'auto'] or isinstance(g, (int, float))]

                # 创建热力图数据
                heatmap_data = []
                for C in C_values:
                    row = []
                    for gamma in gamma_values:
                        mask = (cv_results_df['param_C'] == C) & (cv_results_df['param_gamma'] == gamma)
                        if mask.any():
                            score = cv_results_df.loc[mask, 'mean_test_score'].values[0]
                            row.append(score)
                        else:
                            row.append(np.nan)
                    heatmap_data.append(row)

                # 绘制热力图
                if heatmap_data:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(heatmap_data, annot=True, fmt='.3f',
                                xticklabels=gamma_values, yticklabels=C_values,
                                cmap='viridis', cbar_kws={'label': 'R² Score'})
                    plt.xlabel(plot_config["hyperparameter_heatmap_xlabel"], fontsize=12)
                    plt.ylabel(plot_config["hyperparameter_heatmap_ylabel"], fontsize=12)
                    plt.title(plot_config["hyperparameter_heatmap_title"], fontsize=16, fontweight='bold')

                    heatmap_path = os.path.join(output_dir, 'hyperparameter_heatmap.png')
                    plt.tight_layout()
                    plt.savefig(heatmap_path, dpi=dpi)
                    plt.close()

                    imgs.append({
                        "name": "超参数热力图",
                        "img": heatmap_path,
                        "data": cv_results_path
                    })
        else:
            # 使用手动设置的参数
            svr_model = SVR(**base_params)
            svr_model.fit(X_train, y_train)

        # 7. 模型评估
        y_train_pred = svr_model.predict(X_train)
        y_test_pred = svr_model.predict(X_test)

        # 计算评估指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # 交叉验证
        cv_scores = cross_val_score(svr_model, X_scaled, y, cv=svr_params["cv_folds"], scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # 支持向量信息
        n_support_vectors = len(svr_model.support_vectors_) if hasattr(svr_model, 'support_vectors_') else 0
        support_ratio = n_support_vectors / len(X_train) if len(X_train) > 0 else 0

        print("\n" + "=" * 50)
        print("SVR模型评估结果")
        print("=" * 50)
        print(f"\n训练集:")
        print(f"  R² = {train_r2:.6f}")
        print(f"  RMSE = {train_rmse:.6f}")
        print(f"  MAE = {train_mae:.6f}")
        print(f"  支持向量数量: {n_support_vectors} ({support_ratio:.1%})")

        print(f"\n测试集:")
        print(f"  R² = {test_r2:.6f}")
        print(f"  RMSE = {test_rmse:.6f}")
        print(f"  MAE = {test_mae:.6f}")

        print(f"\n交叉验证 ({svr_params['cv_folds']}折):")
        print(f"  R²均值 = {cv_mean:.6f}")
        print(f"  R²标准差 = {cv_std:.6f}")

        # 8. 生成训练集预测 vs 实际值散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=50, label='训练集样本')

        # 绘制理想线
        min_val = min(y_train.min(), y_train_pred.min())
        max_val = max(y_train.max(), y_train_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='理想线')

        plt.xlabel(plot_config["scatter_train_xlabel"], fontsize=12)
        plt.ylabel(plot_config["scatter_train_ylabel"], fontsize=12)
        plt.title(f"{plot_config['scatter_train_title']}\nR² = {train_r2:.4f}, RMSE = {train_rmse:.2f}",
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # 添加文本标注
        text_str = f'样本数: {len(y_train)}\n特征数: {len(X_columns)}\n核函数: {svr_params["kernel"]}\nC: {svr_params["C"]}'
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        scatter_train_path = os.path.join(output_dir, 'scatter_train.png')
        plt.tight_layout()
        plt.savefig(scatter_train_path, dpi=dpi)
        plt.close()

        # 保存数据
        scatter_train_data_path = os.path.join(output_dir, 'scatter_train_data.txt')
        scatter_train_data = pd.DataFrame({
            'actual': y_train,
            'predicted': y_train_pred
        })
        scatter_train_data.to_csv(scatter_train_data_path, sep='\t', index=False)

        imgs.append({
            "name": "训练集预测散点图",
            "img": scatter_train_path,
            "data": scatter_train_data_path
        })

        # 9. 生成测试集预测 vs 实际值散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_test_pred, alpha=0.6, color='green', s=50, label='测试集样本')

        # 绘制理想线
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='理想线')

        plt.xlabel(plot_config["scatter_test_xlabel"], fontsize=12)
        plt.ylabel(plot_config["scatter_test_ylabel"], fontsize=12)
        plt.title(f"{plot_config['scatter_test_title']}\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}",
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # 添加文本标注
        text_str = f'样本数: {len(y_test)}\n特征数: {len(X_columns)}\n核函数: {svr_params["kernel"]}\nC: {svr_params["C"]}'
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        scatter_test_path = os.path.join(output_dir, 'scatter_test.png')
        plt.tight_layout()
        plt.savefig(scatter_test_path, dpi=dpi)
        plt.close()

        # 保存数据
        scatter_test_data_path = os.path.join(output_dir, 'scatter_test_data.txt')
        scatter_test_data = pd.DataFrame({
            'actual': y_test,
            'predicted': y_test_pred
        })
        scatter_test_data.to_csv(scatter_test_data_path, sep='\t', index=False)

        imgs.append({
            "name": "测试集预测散点图",
            "img": scatter_test_path,
            "data": scatter_test_data_path
        })

        # 10. 生成残差分布图
        residuals = y_test - y_test_pred

        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零残差线')

        # 添加统计信息
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        plt.axvline(x=mean_residual, color='green', linestyle='--', linewidth=2,
                    label=f'均值: {mean_residual:.4f}')

        plt.xlabel(plot_config["residuals_xlabel"], fontsize=12)
        plt.ylabel(plot_config["residuals_ylabel"], fontsize=12)
        plt.title(f"{plot_config['residuals_title']}\n均值: {mean_residual:.4f}, 标准差: {std_residual:.4f}",
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        residuals_path = os.path.join(output_dir, 'residuals.png')
        plt.tight_layout()
        plt.savefig(residuals_path, dpi=dpi)
        plt.close()

        # 保存残差数据
        residuals_data_path = os.path.join(output_dir, 'residuals_data.txt')
        residuals_data = pd.DataFrame({
            'residual': residuals,
            'actual': y_test,
            'predicted': y_test_pred
        })
        residuals_data.to_csv(residuals_data_path, sep='\t', index=False)

        imgs.append({
            "name": "残差分布图",
            "img": residuals_path,
            "data": residuals_data_path
        })

        # 11. 生成Q-Q图（残差正态性检验）
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.xlabel(plot_config["qq_plot_xlabel"], fontsize=12)
        plt.ylabel(plot_config["qq_plot_ylabel"], fontsize=12)
        plt.title(plot_config["qq_plot_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        qq_plot_path = os.path.join(output_dir, 'qq_plot.png')
        plt.tight_layout()
        plt.savefig(qq_plot_path, dpi=dpi)
        plt.close()

        imgs.append({
            "name": "Q-Q图",
            "img": qq_plot_path,
            "data": residuals_data_path
        })

        # 12. 生成残差 vs 拟合值图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_pred, residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

        # 添加epsilon带
        epsilon = svr_params["epsilon"]
        plt.axhline(y=epsilon, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'ε={epsilon}')
        plt.axhline(y=-epsilon, color='green', linestyle='--', linewidth=1, alpha=0.5)

        plt.xlabel(plot_config["residuals_vs_fitted_xlabel"], fontsize=12)
        plt.ylabel(plot_config["residuals_vs_fitted_ylabel"], fontsize=12)
        plt.title(f"{plot_config['residuals_vs_fitted_title']}", fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        residuals_vs_fitted_path = os.path.join(output_dir, 'residuals_vs_fitted.png')
        plt.tight_layout()
        plt.savefig(residuals_vs_fitted_path, dpi=dpi)
        plt.close()

        # 保存数据
        residuals_vs_fitted_data_path = os.path.join(output_dir, 'residuals_vs_fitted_data.txt')
        residuals_vs_fitted_data = pd.DataFrame({
            'fitted': y_test_pred,
            'residual': residuals
        })
        residuals_vs_fitted_data.to_csv(residuals_vs_fitted_data_path, sep='\t', index=False)

        imgs.append({
            "name": "残差 vs 拟合值图",
            "img": residuals_vs_fitted_path,
            "data": residuals_vs_fitted_data_path
        })

        # 13. 生成预测误差图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

        plt.xlabel(plot_config["prediction_error_xlabel"], fontsize=12)
        plt.ylabel(plot_config["prediction_error_ylabel"], fontsize=12)
        plt.title(plot_config["prediction_error_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        prediction_error_path = os.path.join(output_dir, 'prediction_error.png')
        plt.tight_layout()
        plt.savefig(prediction_error_path, dpi=dpi)
        plt.close()

        imgs.append({
            "name": "预测误差图",
            "img": prediction_error_path,
            "data": residuals_data_path
        })

        # 14. 生成支持向量分布图（使用PCA降维到2D）
        if hasattr(svr_model, 'support_vectors_') and len(svr_model.support_vectors_) > 0:
            try:
                # 使用PCA将特征降维到2维
                pca_2d = PCA(n_components=2)
                X_train_2d = pca_2d.fit_transform(X_train)
                support_vectors_2d = pca_2d.transform(svr_model.support_vectors_)

                plt.figure(figsize=(10, 8))

                # 绘制所有训练样本
                plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                            alpha=0.3, s=20, c='gray', label='训练样本')

                # 绘制支持向量
                plt.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                            alpha=0.8, s=100, c='red', marker='s',
                            edgecolors='black', label='支持向量')

                plt.xlabel(plot_config["support_vectors_xlabel"], fontsize=12)
                plt.ylabel(plot_config["support_vectors_ylabel"], fontsize=12)
                plt.title(
                    f"{plot_config['support_vectors_title']}\n支持向量数: {n_support_vectors} ({support_ratio:.1%})",
                    fontsize=14, fontweight='bold')
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)

                support_vectors_path = os.path.join(output_dir, 'support_vectors.png')
                plt.tight_layout()
                plt.savefig(support_vectors_path, dpi=dpi)
                plt.close()

                # 保存数据
                support_vectors_data_path = os.path.join(output_dir, 'support_vectors_data.txt')
                support_vectors_data = pd.DataFrame({
                    'pc1': support_vectors_2d[:, 0],
                    'pc2': support_vectors_2d[:, 1]
                })
                support_vectors_data.to_csv(support_vectors_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "支持向量分布图",
                    "img": support_vectors_path,
                    "data": support_vectors_data_path
                })
            except Exception as e:
                print(f"生成支持向量分布图时出错: {e}")

        # 15. 生成ε-不敏感带示意图（选择一个特征）
        if X.shape[1] >= 1:
            try:
                # 选择一个与目标相关性最高的特征
                feature_idx = 0
                if X.shape[1] > 1:
                    correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
                    feature_idx = np.argmax(np.abs(correlations))

                feature_name = X_columns[feature_idx] if feature_idx < len(X_columns) else f"Feature_{feature_idx}"

                # 创建测试数据
                X_test_single = X_test[:, feature_idx].reshape(-1, 1)
                X_test_full = np.zeros((X_test.shape[0], X.shape[1]))
                X_test_full[:, feature_idx] = X_test_single.flatten()

                # 预测
                y_pred_single = svr_model.predict(X_test_full)

                plt.figure(figsize=(10, 6))

                # 绘制实际值
                plt.scatter(X_test_single, y_test, alpha=0.6, s=30, label='实际值')

                # 绘制预测值
                plt.scatter(X_test_single, y_pred_single, alpha=0.6, s=30, label='预测值')

                # 绘制ε-不敏感带
                epsilon = svr_params["epsilon"]
                plt.fill_between(X_test_single.flatten(),
                                 y_pred_single - epsilon,
                                 y_pred_single + epsilon,
                                 alpha=0.2, color='green', label=f'ε={epsilon} 不敏感带')

                plt.xlabel(plot_config["epsilon_tube_xlabel"] + f" ({feature_name})", fontsize=12)
                plt.ylabel(plot_config["epsilon_tube_ylabel"], fontsize=12)
                plt.title(f"{plot_config['epsilon_tube_title']}\n特征: {feature_name}, ε={epsilon}",
                          fontsize=14, fontweight='bold')
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)

                epsilon_tube_path = os.path.join(output_dir, 'epsilon_tube.png')
                plt.tight_layout()
                plt.savefig(epsilon_tube_path, dpi=dpi)
                plt.close()

                # 保存数据
                epsilon_tube_data_path = os.path.join(output_dir, 'epsilon_tube_data.txt')
                epsilon_tube_data = pd.DataFrame({
                    'feature_value': X_test_single.flatten(),
                    'actual': y_test,
                    'predicted': y_pred_single,
                    'upper_bound': y_pred_single + epsilon,
                    'lower_bound': y_pred_single - epsilon
                })
                epsilon_tube_data.to_csv(epsilon_tube_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "ε-不敏感带示意图",
                    "img": epsilon_tube_path,
                    "data": epsilon_tube_data_path
                })
            except Exception as e:
                print(f"生成ε-不敏感带示意图时出错: {e}")

        # 16. 生成学习曲线
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            test_scores = []

            for size in train_sizes:
                # 选择训练子集
                n_samples = int(size * len(X_train))
                X_subset = X_train[:n_samples]
                y_subset = y_train[:n_samples]

                # 训练模型
                svr_subset = SVR(**base_params)
                svr_subset.fit(X_subset, y_subset)

                # 评估
                train_score = svr_subset.score(X_subset, y_subset)
                test_score = svr_subset.score(X_test, y_test)

                train_scores.append(train_score)
                test_scores.append(test_score)

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes * len(X_train), train_scores, 'o-', linewidth=2,
                     markersize=8, label='训练集得分')
            plt.plot(train_sizes * len(X_train), test_scores, 's-', linewidth=2,
                     markersize=8, label='测试集得分')

            plt.xlabel(plot_config["learning_curve_xlabel"], fontsize=12)
            plt.ylabel(plot_config["learning_curve_ylabel"], fontsize=12)
            plt.title(plot_config["learning_curve_title"], fontsize=16, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            learning_curve_path = os.path.join(output_dir, 'learning_curve.png')
            plt.tight_layout()
            plt.savefig(learning_curve_path, dpi=dpi)
            plt.close()

            # 保存数据
            learning_curve_data_path = os.path.join(output_dir, 'learning_curve_data.txt')
            learning_curve_data = pd.DataFrame({
                'train_size': train_sizes * len(X_train),
                'train_score': train_scores,
                'test_score': test_scores
            })
            learning_curve_data.to_csv(learning_curve_data_path, sep='\t', index=False)

            imgs.append({
                "name": "学习曲线",
                "img": learning_curve_path,
                "data": learning_curve_data_path
            })
        except Exception as e:
            print(f"生成学习曲线时出错: {e}")

        # 17. 生成主成分分析碎石图（如果特征数量较多）
        if len(X_columns) > 2:
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            plt.figure(figsize=(12, 5))

            # 解释方差比例
            plt.subplot(1, 2, 1)
            components = np.arange(1, len(explained_variance_ratio) + 1)
            bars = plt.bar(components, explained_variance_ratio, color='steelblue', alpha=0.7)
            plt.xlabel(plot_config["scree_plot_xlabel"], fontsize=12)
            plt.ylabel(plot_config["scree_plot_ylabel"], fontsize=12)
            plt.title(plot_config["scree_plot_title"], fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')

            # 在柱子上添加数值标签
            for bar, val in zip(bars, explained_variance_ratio):
                if val > 0.01:  # 只显示大于1%的值
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{val:.1%}', ha='center', va='bottom', fontsize=8)

            # 累计解释方差比例
            plt.subplot(1, 2, 2)
            plt.plot(components, cumulative_variance, 'o-', linewidth=2, markersize=8, color='darkorange')
            plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.6, label='95%阈值')
            plt.xlabel(plot_config["cumulative_variance_xlabel"], fontsize=12)
            plt.ylabel(plot_config["cumulative_variance_ylabel"], fontsize=12)
            plt.title(plot_config["cumulative_variance_title"], fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 标记达到95%的组件数
            if cumulative_variance[-1] >= 0.95:
                idx = np.where(cumulative_variance >= 0.95)[0][0]
                plt.axvline(x=components[idx], color='g', linestyle=':', alpha=0.6)
                plt.text(components[idx], 0.5, f'主成分{components[idx]}\n达到95%',
                         ha='right', va='center', fontsize=9, color='g')

            scree_img_path = os.path.join(output_dir, 'pca_scree_plot.png')
            plt.tight_layout()
            plt.savefig(scree_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            scree_data_path = os.path.join(output_dir, 'pca_scree_plot_data.txt')
            scree_data = pd.DataFrame({
                'component': components,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance
            })
            scree_data.to_csv(scree_data_path, sep='\t', index=False)

            imgs.append({
                "name": "主成分分析碎石图",
                "img": scree_img_path,
                "data": scree_data_path
            })

        # 18. 生成特征相关性热图（如果特征数量适中）
        if len(X_columns) <= 20:
            corr_matrix = X.corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.xlabel(plot_config["feature_correlation_xlabel"], fontsize=12)
            plt.ylabel(plot_config["feature_correlation_ylabel"], fontsize=12)
            plt.title(plot_config["feature_correlation_title"], fontsize=16, fontweight='bold')

            corr_img_path = os.path.join(output_dir, 'feature_correlation.png')
            plt.tight_layout()
            plt.savefig(corr_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            corr_data_path = os.path.join(output_dir, 'feature_correlation_data.txt')
            corr_matrix.to_csv(corr_data_path, sep='\t')

            imgs.append({
                "name": "特征相关性热图",
                "img": corr_img_path,
                "data": corr_data_path
            })

        # 19. 保存模型
        model_path = os.path.join(output_dir, 'svr_model.pkl')
        joblib.dump(svr_model, model_path)

        # 20. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": df.shape[0],
                "features": len(X_columns),
                "target_variable": y_column,
                "feature_variables": X_columns
            },
            "model_params": svr_params,
            "performance": {
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse),
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "cv_mean_r2": float(cv_mean),
                "cv_std_r2": float(cv_std),
                "n_support_vectors": n_support_vectors,
                "support_ratio": float(support_ratio)
            },
            "model_file": os.path.basename(model_path),
            "scaler_file": os.path.basename(scaler_path) if svr_params["standardize"] else None
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 21. 生成结果字符串
        result_str += "=======================\n"
        result_str += "SVR模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {len(X_columns)} 个特征\n"
        result_str += f"2. 目标变量: {y_column}\n"
        result_str += f"3. 核函数: {svr_params['kernel']}\n"
        result_str += f"4. 训练集R²: {train_r2:.6f}\n"
        result_str += f"5. 测试集R²: {test_r2:.6f}\n"
        result_str += f"6. 训练集RMSE: {train_rmse:.6f}\n"
        result_str += f"7. 测试集RMSE: {test_rmse:.6f}\n"
        result_str += f"8. 训练集MAE: {train_mae:.6f}\n"
        result_str += f"9. 测试集MAE: {test_mae:.6f}\n"
        result_str += f"10. 交叉验证R²均值: {cv_mean:.6f} (±{cv_std:.6f})\n"
        result_str += f"11. 支持向量数量: {n_support_vectors} ({support_ratio:.1%})\n"

        # 模型性能评估
        if test_r2 > 0.9:
            result_str += "12. 模型性能: 优秀 (R² > 0.9)\n"
        elif test_r2 > 0.7:
            result_str += "12. 模型性能: 良好 (0.7 < R² ≤ 0.9)\n"
        elif test_r2 > 0.5:
            result_str += "12. 模型性能: 一般 (0.5 < R² ≤ 0.7)\n"
        else:
            result_str += "12. 模型性能: 较差 (R² ≤ 0.5)\n"

        # 添加残差统计信息
        result_str += f"\n=== 残差统计信息 ===\n"
        result_str += f"均值: {mean_residual:.4f}\n"
        result_str += f"标准差: {std_residual:.4f}\n"
        result_str += f"偏度: {stats.skew(residuals):.4f}\n"
        result_str += f"峰度: {stats.kurtosis(residuals):.4f}\n"

        # 添加生成的图片信息
        result_str += f"\n=== 生成的图片 ===\n"
        for img_info in imgs:
            img_filename = os.path.basename(img_info['img'])
            data_filename = os.path.basename(img_info['data']) if img_info['data'] else "无数据文件"
            result_str += f"- {img_info['name']}: {img_filename} (数据文件: {data_filename})\n"

        # 添加模型信息文件名
        result_str += f"\n=== 模型文件 ===\n"
        model_filename = os.path.basename(model_path) if 'model_path' in locals() else "未生成"
        result_str += f"SVR模型文件: {model_filename}\n"
        if svr_params["standardize"] and 'scaler_path' in locals():
            scaler_filename = os.path.basename(scaler_path)
            result_str += f"标准化器文件: {scaler_filename}\n"

        # 添加计算使用的参数信息
        result_str += f"\n=== 计算使用的参数信息 ===\n"
        result_str += f"SVR算法参数:\n"

        param_descriptions = {
            "target_column": f"目标列名 (当前: {svr_params['target_column']})",
            "test_size": f"测试集比例 (当前: {svr_params['test_size']})",
            "random_state": f"随机种子 (当前: {svr_params['random_state']})",
            "cv_folds": f"交叉验证折数 (当前: {svr_params['cv_folds']})",
            "standardize": f"是否标准化特征 (当前: {svr_params['standardize']})",
            "kernel": f"核函数类型 (当前: {svr_params['kernel']})",
            "C": f"正则化参数 (当前: {svr_params['C']})",
            "epsilon": f"epsilon值 (当前: {svr_params['epsilon']})",
            "gamma": f"核函数系数 (当前: {svr_params['gamma']})",
            "degree": f"多项式度 (当前: {svr_params['degree']})",
            "coef0": f"核函数独立项 (当前: {svr_params['coef0']})",
            "grid_search": f"是否网格搜索 (当前: {svr_params['grid_search']})",
            "cache_size": f"核缓存大小 (当前: {svr_params['cache_size']})",
            "max_iter": f"最大迭代次数 (当前: {svr_params['max_iter']})",
            "tol": f"收敛容差 (当前: {svr_params['tol']})",
            "shrinking": f"是否使用收缩启发式 (当前: {svr_params['shrinking']})"
        }

        for key, value in svr_params.items():
            if key in param_descriptions:
                result_str += f"  {key}: {value}\n"
                result_str += f"    说明: {param_descriptions[key]}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"SVR模型运行失败: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        result_str = "=======================\n"
        result_str += "SVR模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 预测模型 SVR 运行结束 <<<<<<<<<<<<<<<<<<<<")
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
            "model": {
                "data": model_path if 'model_path' in locals() else ""
            },
            "scaler": {
                "data": scaler_path if 'scaler_path' in locals() else ""
            }
        },
        "start": start_timestamp,
        "end": end_timestamp,
        "expend": expend_time
    }

    # 将结果转换为JSON字符串返回
    return json.dumps(result_json, ensure_ascii=False, indent=2)