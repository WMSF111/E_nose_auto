import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import os
import json
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 预测模型 MLR 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"mlr_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """MLR参数配置"""
        # 1. 扩展算法参数
        mlr_params = {
            "target_column": "target",
            # 目标列名
            # 默认值：'target'

            "test_size": 0.2,
            # 测试集比例
            # 取值范围：0到1之间的浮点数
            # 默认值：0.2

            "random_state": 42,
            # 随机种子
            # 取值范围：正整数或None
            # 默认值：42

            "cv_folds": 5,
            # 交叉验证折数
            # 取值范围：正整数，通常3-10
            # 默认值：5

            "standardize": True,
            # 是否标准化特征
            # 取值范围：布尔值True或False
            # 默认值：True

            "fit_intercept": True,
            # 是否计算截距
            # 取值范围：布尔值True或False
            # 默认值：True

            "positive": False,
            # 是否强制系数为正
            # 取值范围：布尔值True或False
            # 默认值：False

            "n_jobs": None,
            # 并行任务数
            # 取值范围：正整数或None
            # 默认值：None
        }

        # 从传入参数中更新MLR参数
        in_params = params.get("params", {})

        # 更新参数 - 处理可能的列表值
        if "target_column" in in_params:
            mlr_params["target_column"] = in_params["target_column"]

        if "test_size" in in_params:
            test_val = in_params["test_size"]
            mlr_params["test_size"] = AlgModelParameters.format_to_float(test_val)

        if "random_state" in in_params:
            random_val = in_params["random_state"]
            mlr_params["random_state"] = AlgModelParameters.format_to_int(random_val)

        if "cv_folds" in in_params:
            cv_val = in_params["cv_folds"]
            mlr_params["cv_folds"] = AlgModelParameters.format_to_int(cv_val)

        if "standardize" in in_params:
            mlr_params["standardize"] = in_params["standardize"]

        if "fit_intercept" in in_params:
            mlr_params["fit_intercept"] = in_params["fit_intercept"]

        if "positive" in in_params:
            mlr_params["positive"] = in_params["fit_intercept"]

        if "n_jobs" in in_params:
            job_val = in_params["n_jobs"]
            mlr_params["n_jobs"] = AlgModelParameters.format_to_int(job_val)

        print(f"算法参数: {mlr_params}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("预测模型","MLR").copy()

        # 如果用户提供了自定义图片参数，则更新
        if "image_param" in params and isinstance(params["image_param"], dict):
            plot_config.update(params["image_param"])

        # 3. 数据准备
        # 确定目标列
        y_column = mlr_params["target_column"]

        # 检查目标列是否存在
        if y_column not in df.columns:
            error_msg = f"错误: 数据框中不存在目标列 '{y_column}'。请检查数据框列名: {list(df.columns)}"
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
        if mlr_params["standardize"]:
            print("\n标准化特征数据...")
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
        else:
            X_scaled = X.values

        # 5. 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=mlr_params["test_size"],
            random_state=mlr_params["random_state"]
        )

        print(f"\n数据集分割:")
        print(f"  训练集大小: {X_train.shape[0]} ({1 - mlr_params['test_size']:.0%})")
        print(f"  测试集大小: {X_test.shape[0]} ({mlr_params['test_size']:.0%})")

        # 6. 训练多元线性回归模型
        print(f"\n训练多元线性回归模型...")
        mlr_model = LinearRegression(
            fit_intercept=mlr_params["fit_intercept"],
            positive=mlr_params["positive"],
            n_jobs=mlr_params["n_jobs"]
        )

        mlr_model.fit(X_train, y_train)

        # 7. 模型评估
        y_train_pred = mlr_model.predict(X_train)
        y_test_pred = mlr_model.predict(X_test)

        # 计算评估指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # 交叉验证
        cv_scores = cross_val_score(mlr_model, X_scaled, y, cv=mlr_params["cv_folds"], scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        print("\n" + "=" * 50)
        print("MLR模型评估结果")
        print("=" * 50)
        print(f"\n训练集:")
        print(f"  R² = {train_r2:.6f}")
        print(f"  RMSE = {train_rmse:.6f}")
        print(f"  MAE = {train_mae:.6f}")

        print(f"\n测试集:")
        print(f"  R² = {test_r2:.6f}")
        print(f"  RMSE = {test_rmse:.6f}")
        print(f"  MAE = {test_mae:.6f}")

        print(f"\n交叉验证 ({mlr_params['cv_folds']}折):")
        print(f"  R²均值 = {cv_mean:.6f}")
        print(f"  R²标准差 = {cv_std:.6f}")

        # 8. 模型系数
        coefficients_df = pd.DataFrame({
            'feature': X_columns,
            'coefficient': mlr_model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        print(f"\n模型系数 (前10个最重要的特征):")
        print(coefficients_df.head(10).to_string(index=False))

        # 9. 生成训练集预测 vs 实际值散点图
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
        text_str = f'样本数: {len(y_train)}\n特征数: {len(X_columns)}\n截距: {mlr_model.intercept_:.4f}'
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

        # 10. 生成测试集预测 vs 实际值散点图
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
        text_str = f'样本数: {len(y_test)}\n特征数: {len(X_columns)}\n截距: {mlr_model.intercept_:.4f}'
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

        # 11. 生成模型系数图
        plt.figure(figsize=(12, 6))

        # 选择前15个最重要的系数
        top_n_coef = min(15, len(coefficients_df))
        top_coef = coefficients_df.head(top_n_coef)

        colors = ['red' if coef > 0 else 'blue' for coef in top_coef['coefficient']]

        plt.barh(range(len(top_coef)), top_coef['coefficient'][::-1], color=colors[::-1])
        plt.yticks(range(len(top_coef)), top_coef['feature'][::-1])

        # 添加零线
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        plt.xlabel(plot_config["coefficients_ylabel"], fontsize=12)
        plt.ylabel(plot_config["coefficients_xlabel"], fontsize=12)
        plt.title(plot_config["coefficients_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        coef_plot_path = os.path.join(output_dir, 'coefficients.png')
        plt.tight_layout()
        plt.savefig(coef_plot_path, dpi=dpi)
        plt.close()

        # 保存系数数据
        coef_data_path = os.path.join(output_dir, 'coefficients_data.txt')
        coefficients_df.to_csv(coef_data_path, sep='\t', index=False)

        imgs.append({
            "name": "模型系数图",
            "img": coef_plot_path,
            "data": coef_data_path
        })

        # 12. 生成残差分布图
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

        # 13. 生成Q-Q图（残差正态性检验）
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
            "data": residuals_data_path  # 使用相同的残差数据
        })

        # 14. 生成残差 vs 拟合值图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_pred, residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

        plt.xlabel(plot_config["residuals_vs_fitted_xlabel"], fontsize=12)
        plt.ylabel(plot_config["residuals_vs_fitted_ylabel"], fontsize=12)
        plt.title(plot_config["residuals_vs_fitted_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

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

        # 15. 生成预测误差图
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
            "data": residuals_data_path  # 使用相同的残差数据
        })

        # 16. 生成主成分分析碎石图（如果特征数量较多）
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

        # 17. 生成特征相关性热图（如果特征数量适中）
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

        # 18. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": df.shape[0],
                "features": len(X_columns),
                "target_variable": y_column,
                "feature_variables": X_columns
            },
            "model_params": mlr_params,
            "performance": {
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse),
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "cv_mean_r2": float(cv_mean),
                "cv_std_r2": float(cv_std),
                "intercept": float(mlr_model.intercept_)
            },
            "coefficients": coefficients_df.to_dict('records'),
            "residuals_statistics": {
                "mean": float(mean_residual),
                "std": float(std_residual),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals))
            }
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 19. 生成结果字符串
        result_str += "=======================\n"
        result_str += "MLR模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {len(X_columns)} 个特征\n"
        result_str += f"2. 目标变量: {y_column}\n"
        result_str += f"3. 训练集R²: {train_r2:.6f}\n"
        result_str += f"4. 测试集R²: {test_r2:.6f}\n"
        result_str += f"5. 训练集RMSE: {train_rmse:.6f}\n"
        result_str += f"6. 测试集RMSE: {test_rmse:.6f}\n"
        result_str += f"7. 训练集MAE: {train_mae:.6f}\n"
        result_str += f"8. 测试集MAE: {test_mae:.6f}\n"
        result_str += f"9. 交叉验证R²均值: {cv_mean:.6f} (±{cv_std:.6f})\n"

        # 模型性能评估
        if test_r2 > 0.9:
            result_str += "10. 模型性能: 优秀 (R² > 0.9)\n"
        elif test_r2 > 0.7:
            result_str += "10. 模型性能: 良好 (0.7 < R² ≤ 0.9)\n"
        elif test_r2 > 0.5:
            result_str += "10. 模型性能: 一般 (0.5 < R² ≤ 0.7)\n"
        else:
            result_str += "10. 模型性能: 较差 (R² ≤ 0.5)\n"

        # 添加模型系数信息
        result_str += f"\n=== 最重要的5个特征 ===\n"
        for i, row in coefficients_df.head(5).iterrows():
            result_str += f"  {row['feature']}: 系数 = {row['coefficient']:.4f}\n"

        result_str += f"\n截距: {mlr_model.intercept_:.4f}\n"

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
        result_str += f"\n=== 模型信息文件 ===\n"
        model_info_filename = os.path.basename(model_info_path) if 'model_info_path' in locals() else "未生成"
        result_str += f"模型信息文件: {model_info_filename}\n"

        # 添加计算使用的参数信息
        result_str += f"\n=== 计算使用的参数信息 ===\n"
        result_str += f"MLR算法参数:\n"

        param_descriptions = {
            "target_column": f"目标列名 (当前: {mlr_params['target_column']})",
            "test_size": f"测试集比例 (当前: {mlr_params['test_size']})",
            "random_state": f"随机种子 (当前: {mlr_params['random_state']})",
            "cv_folds": f"交叉验证折数 (当前: {mlr_params['cv_folds']})",
            "standardize": f"是否标准化特征 (当前: {mlr_params['standardize']})",
            "fit_intercept": f"是否计算截距 (当前: {mlr_params['fit_intercept']})",
            "positive": f"是否强制系数为正 (当前: {mlr_params['positive']})",
            "n_jobs": f"并行任务数 (当前: {mlr_params['n_jobs']})"
        }

        for key, value in mlr_params.items():
            if key in param_descriptions:
                result_str += f"  {key}: {value}\n"
                result_str += f"    说明: {param_descriptions[key]}\n"
            else:
                result_str += f"  {key}: {value}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"MLR模型运行失败: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        result_str = "=======================\n"
        result_str += "MLR模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 预测模型 MLR 运行结束 <<<<<<<<<<<<<<<<<<<<")
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
            "coefficients": {
                "data": coef_data_path if 'coef_data_path' in locals() else ""
            },
            "residuals": {
                "data": residuals_data_path if 'residuals_data_path' in locals() else ""
            }
        },
        "start": start_timestamp,
        "end": end_timestamp,
        "expend": expend_time
    }

    # 将结果转换为JSON字符串返回
    return json.dumps(result_json, ensure_ascii=False, indent=2)