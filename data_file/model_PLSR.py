import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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
    print(">>>>>>>>>>>>>>>>>>>> 预测模型 PLSR 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"plsr_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """PLSR参数配置 - 带详细注释"""
        # 1. 扩展算法参数
        plsr_params = {
            "target_sensor": "target",
            # 目标列名
            # 含义：要预测的目标变量列名
            # 默认值：'target'

            "n_components": "auto",
            # PLS成分数
            # 含义：要保留的PLS成分数
            # 可选值：
            # - 'auto': 通过交叉验证自动选择最佳成分数（默认）
            # - 正整数：手动指定成分数
            # 取值范围：1到特征数之间的整数
            # 默认值：'auto'

            "scale": True,
            # 是否标准化
            # 含义：是否在PLS模型内部对数据进行标准化
            # 取值范围：布尔值True或False
            # 默认值：True

            "max_iter": 500,
            # 最大迭代次数
            # 含义：NIPALS算法的最大迭代次数
            # 取值范围：正整数，通常100-1000
            # 默认值：500

            "tol": 1e-6,
            # 收敛阈值
            # 含义：算法收敛的阈值
            # 取值范围：正浮点数，通常1e-9到1e-4
            # 默认值：1e-6

            "copy": True,
            # 是否复制数据
            # 含义：是否复制X和Y，如果为False则可能覆盖原始数据
            # 取值范围：布尔值True或False
            # 默认值：True

            "test_size": 0.2,
            # 测试集比例
            # 含义：测试集占总数据的比例
            # 取值范围：0到1之间的浮点数
            # 默认值：0.2

            "random_state": 42,
            # 随机种子
            # 含义：控制随机数生成，确保结果可复现
            # 取值范围：正整数或None
            # 默认值：42

            "cv_folds": 5,
            # 交叉验证折数
            # 含义：用于选择最佳成分数的交叉验证折数
            # 取值范围：正整数，通常3-10
            # 默认值：5

            "max_components_to_try": 10,
            # 最大尝试成分数
            # 含义：当n_components='auto'时，尝试的最大成分数
            # 取值范围：正整数，通常5-20
            # 默认值：10

            "vip_threshold": 1.0,
            # VIP分数阈值
            # 含义：用于判断特征重要性的VIP分数阈值
            # 取值范围：正浮点数，通常0.8-1.5
            # 默认值：1.0
        }

        # 从传入参数中更新PLSR参数
        in_params = params.get("params", {})

        # 更新参数
        if "target_sensor" in in_params:
            plsr_params["target_sensor"] = in_params["target_sensor"]
        # if "n_components" in in_params and in_params["n_components"] != "auto":
        #     plsr_params["n_components"] = AlgModelParameters.format_to_int(in_params["n_components"])
        if "scale" in in_params:
            plsr_params["scale"] = in_params["scale"]
        if "max_iter" in in_params:
            plsr_params["max_iter"] = AlgModelParameters.format_to_int(in_params["max_iter"])
        if "tol" in in_params:
            plsr_params["tol"] = AlgModelParameters.format_to_float(in_params["tol"])
        if "test_size" in in_params:
            plsr_params["test_size"] = AlgModelParameters.format_to_float(in_params["test_size"])
        if "random_state" in in_params:
            plsr_params["random_state"] = AlgModelParameters.format_to_int(in_params["random_state"])
        if "cv_folds" in in_params:
            plsr_params["cv_folds"] = AlgModelParameters.format_to_int(in_params["cv_folds"])
        if "max_components_to_try" in in_params:
            plsr_params["max_components_to_try"] = AlgModelParameters.format_to_int(in_params["max_components_to_try"])
        if "vip_threshold" in in_params:
            plsr_params["vip_threshold"] = AlgModelParameters.format_to_float(in_params["vip_threshold"])

        print(f"算法参数: {plsr_params}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("预测模型", "PLSR").copy()

        # 如果用户提供了自定义图片参数，则更新
        if "image_param" in params and isinstance(params["image_param"], dict):
            plot_config.update(params["image_param"])

        # 3. 数据准备
        # 确定目标列
        y_column = plsr_params["target_sensor"]

        # 检查目标列是否存在
        if y_column not in df.columns:
            error_msg = f"错误: 数据框中不存在目标列 '{y_column}'。请检查数据框列名: {list(df.columns)}"
            raise ValueError(error_msg)

        # 确定特征列 - 除了目标列之外的所有列
        X_columns = [col for col in df.columns if col != y_column and col != "target"]

        # 检查特征列数量
        if len(X_columns) <= 1:
            error_msg = f"错误: 特征列数量不足。目标列 '{y_column}' 之外的特征列只有 {len(X_columns)} 个，至少需要 2 个特征列进行建模。"
            raise ValueError(error_msg)

        X = df[X_columns]
        y = df[y_column]

        print(f"\n数据信息:")
        print(f"  目标变量: {y_column}")
        print(f"  特征数量: {len(X_columns)}")
        print(f"  特征列: {X_columns}")
        print(f"  总样本数: {df.shape[0]}")
        print(f"  特征数据形状: {X.shape}")
        print(f"  目标数据形状: {y.shape}")

        # 移除包含NaN的行
        data_combined = pd.concat([X, y], axis=1)
        data_cleaned = data_combined.dropna()
        if len(data_cleaned) < len(data_combined):
            print(f"警告: 移除了 {len(data_combined) - len(data_cleaned)} 个包含NaN的行")
            X = data_cleaned[X_columns]
            y = data_cleaned[y_column]

        # 4. 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        y_scaled_flat = y_scaled.ravel()

        # 5. 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled_flat,
            test_size=plsr_params["test_size"],
            random_state=plsr_params["random_state"]
        )

        print(f"\n数据集分割:")
        print(f"  训练集大小: {X_train.shape[0]} ({1 - plsr_params['test_size']:.0%})")
        print(f"  测试集大小: {X_test.shape[0]} ({plsr_params['test_size']:.0%})")

        # 6. 确定最佳成分数（如果设置为auto）
        n_components = plsr_params["n_components"]
        if n_components == "auto":
            print("\n通过交叉验证确定最佳PLS成分数...")
            max_components = min(plsr_params["max_components_to_try"], X.shape[1])
            n_components_range = range(1, max_components + 1)

            mse_scores = []
            r2_scores = []

            for n_comp in n_components_range:
                pls = PLSRegression(n_components=n_comp,
                                    scale=plsr_params["scale"],
                                    max_iter=plsr_params["max_iter"],
                                    tol=plsr_params["tol"])

                # 使用交叉验证
                y_cv = cross_val_predict(pls, X_train, y_train, cv=plsr_params["cv_folds"])
                mse = mean_squared_error(y_train, y_cv)
                r2 = r2_score(y_train, y_cv)
                mse_scores.append(mse)
                r2_scores.append(r2)
                print(f"  成分数 {n_comp:2d}: MSE = {mse:.6f}, R² = {r2:.6f}")

            # 选择最佳成分数（最小MSE）
            best_idx = np.argmin(mse_scores)
            n_components = n_components_range[best_idx]
            print(f"\n最佳PLS成分数: {n_components}")
            print(f"  对应MSE: {mse_scores[best_idx]:.6f}")
            print(f"  对应R²: {r2_scores[best_idx]:.6f}")

            # 生成碎石图
            plt.figure(figsize=(10, 6))
            plt.plot(n_components_range, mse_scores, 'bo-', linewidth=2, markersize=8, label='交叉验证MSE')
            plt.plot(n_components_range, r2_scores, 'rs-', linewidth=2, markersize=8, label='交叉验证R²')

            # 标记最佳成分数
            plt.axvline(x=n_components, color='g', linestyle='--', linewidth=2, alpha=0.7)
            plt.text(n_components + 0.1, max(mse_scores) * 0.9, f'最佳成分数: {n_components}',
                     fontsize=12, color='g', fontweight='bold')

            plt.xlabel(plot_config["scree_plot_xlabel"], fontsize=12)
            plt.ylabel(plot_config["scree_plot_ylabel"], fontsize=12)
            plt.title(plot_config["scree_plot_title"], fontsize=16, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.xticks(n_components_range)

            scree_plot_path = os.path.join(output_dir, 'scree_plot.png')
            plt.tight_layout()
            plt.savefig(scree_plot_path, dpi=dpi)
            plt.close()

            # 保存数据
            scree_data_path = os.path.join(output_dir, 'scree_plot_data.txt')
            scree_data = pd.DataFrame({
                'n_components': list(n_components_range),
                'mse': mse_scores,
                'r2': r2_scores
            })
            scree_data.to_csv(scree_data_path, sep='\t', index=False)

            imgs.append({
                "name": "PLS碎石图",
                "img": scree_plot_path,
                "data": scree_data_path
            })
        else:
            n_components = int(n_components)
            print(f"\n使用指定的PLS成分数: {n_components}")

        # 7. 训练最终模型
        print(f"\n使用成分数({n_components})训练PLS模型...")
        pls_model = PLSRegression(
            n_components=n_components,
            scale=plsr_params["scale"],
            max_iter=plsr_params["max_iter"],
            tol=plsr_params["tol"],
            copy=plsr_params["copy"]
        )

        pls_model.fit(X_train, y_train)

        # 8. 模型评估
        y_train_pred = pls_model.predict(X_train)
        y_test_pred = pls_model.predict(X_test)

        # 确保预测值是2D数组
        if y_train_pred.ndim == 1:
            y_train_pred = y_train_pred.reshape(-1, 1)
        if y_test_pred.ndim == 1:
            y_test_pred = y_test_pred.reshape(-1, 1)

        # 将数据转换回原始尺度
        y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_train_pred_original = scaler_y.inverse_transform(y_train_pred)
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        y_test_pred_original = scaler_y.inverse_transform(y_test_pred)

        # 计算评估指标
        train_r2 = r2_score(y_train_original, y_train_pred_original)
        test_r2 = r2_score(y_test_original, y_test_pred_original)
        train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
        test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
        train_mae = mean_absolute_error(y_train_original, y_train_pred_original)
        test_mae = mean_absolute_error(y_test_original, y_test_pred_original)

        print("\n" + "=" * 50)
        print("PLSR模型评估结果")
        print("=" * 50)
        print(f"\n训练集:")
        print(f"  R² = {train_r2:.6f}")
        print(f"  RMSE = {train_rmse:.6f}")
        print(f"  MAE = {train_mae:.6f}")

        print(f"\n测试集:")
        print(f"  R² = {test_r2:.6f}")
        print(f"  RMSE = {test_rmse:.6f}")
        print(f"  MAE = {test_mae:.6f}")

        # 9. 计算VIP分数
        print("\n计算特征重要性(VIP分数)...")
        # 获取模型内部参数
        T = pls_model.x_scores_  # 得分矩阵
        W = pls_model.x_weights_  # 权重矩阵
        Q = pls_model.y_loadings_  # Y载荷

        # 计算VIP分数
        p = X.shape[1]
        vip_scores = np.zeros((p,))

        for i in range(p):
            numerator = np.sum((W[i, :] ** 2) * np.sum(Q ** 2, axis=0))
            denominator = np.sum(np.sum(Q ** 2, axis=0))
            vip_scores[i] = np.sqrt(p * numerator / denominator)

        vip_df = pd.DataFrame({
            'feature': X_columns,
            'vip_score': vip_scores
        }).sort_values('vip_score', ascending=False)

        # 10. 模型系数
        coefficients_df = pd.DataFrame({
            'feature': X_columns,
            'coefficient': pls_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)

        # 11. 生成训练集预测 vs 实际值散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(y_train_original, y_train_pred_original, alpha=0.6, color='blue', s=50, label='训练集样本')

        # 绘制理想线
        min_val = min(y_train_original.min(), y_train_pred_original.min())
        max_val = max(y_train_original.max(), y_train_pred_original.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='理想线')

        plt.xlabel(plot_config["scatter_train_xlabel"], fontsize=12)
        plt.ylabel(plot_config["scatter_train_ylabel"], fontsize=12)
        plt.title(f"{plot_config['scatter_train_title']}\nR² = {train_r2:.4f}, RMSE = {train_rmse:.2f}",
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # 添加文本标注
        text_str = f'样本数: {len(y_train_original)}\nPLS成分数: {n_components}'
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
            'actual': y_train_original.flatten(),
            'predicted': y_train_pred_original.flatten()
        })
        scatter_train_data.to_csv(scatter_train_data_path, sep='\t', index=False)

        imgs.append({
            "name": "训练集预测散点图",
            "img": scatter_train_path,
            "data": scatter_train_data_path
        })

        # 12. 生成测试集预测 vs 实际值散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test_original, y_test_pred_original, alpha=0.6, color='green', s=50, label='测试集样本')

        # 绘制理想线
        min_val = min(y_test_original.min(), y_test_pred_original.min())
        max_val = max(y_test_original.max(), y_test_pred_original.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='理想线')

        plt.xlabel(plot_config["scatter_test_xlabel"], fontsize=12)
        plt.ylabel(plot_config["scatter_test_ylabel"], fontsize=12)
        plt.title(f"{plot_config['scatter_test_title']}\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}",
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # 添加文本标注
        text_str = f'样本数: {len(y_test_original)}\nPLS成分数: {n_components}'
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
            'actual': y_test_original.flatten(),
            'predicted': y_test_pred_original.flatten()
        })
        scatter_test_data.to_csv(scatter_test_data_path, sep='\t', index=False)

        imgs.append({
            "name": "测试集预测散点图",
            "img": scatter_test_path,
            "data": scatter_test_data_path
        })

        # 13. 生成VIP分数图
        plt.figure(figsize=(12, 6))

        # 选择前15个最重要的特征
        top_n = min(15, len(vip_df))
        top_vip = vip_df.head(top_n)

        colors = ['red' if score > plsr_params["vip_threshold"] else 'blue' for score in top_vip['vip_score']]

        plt.barh(range(len(top_vip)), top_vip['vip_score'][::-1], color=colors[::-1])
        plt.yticks(range(len(top_vip)), top_vip['feature'][::-1])

        # 添加阈值线
        plt.axvline(x=plsr_params["vip_threshold"], color='red', linestyle='--', linewidth=2,
                    alpha=0.7, label=f'VIP阈值 ({plsr_params["vip_threshold"]})')

        plt.xlabel(plot_config["vip_scores_ylabel"], fontsize=12)
        plt.ylabel(plot_config["vip_scores_xlabel"], fontsize=12)
        plt.title(plot_config["vip_scores_title"], fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='x')

        vip_plot_path = os.path.join(output_dir, 'vip_scores.png')
        plt.tight_layout()
        plt.savefig(vip_plot_path, dpi=dpi)
        plt.close()

        # 保存VIP数据
        vip_data_path = os.path.join(output_dir, 'vip_scores_data.txt')
        vip_df.to_csv(vip_data_path, sep='\t', index=False)

        imgs.append({
            "name": "VIP分数图",
            "img": vip_plot_path,
            "data": vip_data_path
        })

        # 14. 生成系数图
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

        # 15. 生成残差分布图
        residuals = y_test_original.flatten() - y_test_pred_original.flatten()

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
            'actual': y_test_original.flatten(),
            'predicted': y_test_pred_original.flatten()
        })
        residuals_data.to_csv(residuals_data_path, sep='\t', index=False)

        imgs.append({
            "name": "残差分布图",
            "img": residuals_path,
            "data": residuals_data_path
        })

        # 16. 生成Q-Q图（残差正态性检验）
        from scipy import stats

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

        # 17. 生成实际值 vs 预测值时序图
        plt.figure(figsize=(14, 6))

        # 对测试集样本排序
        sorted_indices = np.argsort(y_test_original.flatten())
        y_test_sorted = y_test_original.flatten()[sorted_indices]
        y_test_pred_sorted = y_test_pred_original.flatten()[sorted_indices]

        plt.plot(range(len(y_test_sorted)), y_test_sorted, 'b-', linewidth=2, label='实际值')
        plt.plot(range(len(y_test_pred_sorted)), y_test_pred_sorted, 'r--', linewidth=2, label='预测值')

        plt.xlabel(plot_config["actual_vs_pred_xlabel"], fontsize=12)
        plt.ylabel(plot_config["actual_vs_pred_ylabel"], fontsize=12)
        plt.title(plot_config["actual_vs_pred_title"], fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        actual_vs_pred_path = os.path.join(output_dir, 'actual_vs_pred.png')
        plt.tight_layout()
        plt.savefig(actual_vs_pred_path, dpi=dpi)
        plt.close()

        # 保存数据
        actual_vs_pred_data_path = os.path.join(output_dir, 'actual_vs_pred_data.txt')
        actual_vs_pred_data = pd.DataFrame({
            'index': range(len(y_test_sorted)),
            'actual': y_test_sorted,
            'predicted': y_test_pred_sorted
        })
        actual_vs_pred_data.to_csv(actual_vs_pred_data_path, sep='\t', index=False)

        imgs.append({
            "name": "实际值 vs 预测值时序图",
            "img": actual_vs_pred_path,
            "data": actual_vs_pred_data_path
        })

        # 18. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": df.shape[0],
                "features": len(X_columns),
                "target_variable": y_column,
                "feature_variables": X_columns
            },
            "model_params": plsr_params,
            "performance": {
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse),
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "n_components": int(n_components),
                "n_iterations": pls_model.n_iter_ if hasattr(pls_model, 'n_iter_') else None
            },
            "vip_scores": vip_df.to_dict('records'),
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
        result_str += "PLSR模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {len(X_columns)} 个特征\n"
        result_str += f"2. 目标变量: {y_column}\n"
        result_str += f"3. PLS成分数: {n_components}\n"
        result_str += f"4. 训练集R²: {train_r2:.6f}\n"
        result_str += f"5. 测试集R²: {test_r2:.6f}\n"
        result_str += f"6. 训练集RMSE: {train_rmse:.6f}\n"
        result_str += f"7. 测试集RMSE: {test_rmse:.6f}\n"
        result_str += f"8. 训练集MAE: {train_mae:.6f}\n"
        result_str += f"9. 测试集MAE: {test_mae:.6f}\n"

        # 模型性能评估
        if test_r2 > 0.9:
            result_str += "10. 模型性能: 优秀 (R² > 0.9)\n"
        elif test_r2 > 0.7:
            result_str += "10. 模型性能: 良好 (0.7 < R² ≤ 0.9)\n"
        elif test_r2 > 0.5:
            result_str += "10. 模型性能: 一般 (0.5 < R² ≤ 0.7)\n"
        else:
            result_str += "10. 模型性能: 较差 (R² ≤ 0.5)\n"

        # 添加VIP分数分析
        result_str += f"\n=== 特征重要性分析 (VIP分数) ===\n"
        result_str += f"VIP分数阈值: {plsr_params['vip_threshold']}\n"
        important_features = vip_df[vip_df['vip_score'] > plsr_params['vip_threshold']]
        result_str += f"重要特征数量 (VIP > {plsr_params['vip_threshold']}): {len(important_features)}\n"

        if len(important_features) > 0:
            result_str += f"前5个重要特征:\n"
            for i, row in important_features.head().iterrows():
                result_str += f"  {row['feature']}: VIP分数 = {row['vip_score']:.4f}\n"

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
        result_str += f"PLSR算法参数:\n"

        # 参数详细说明
        param_descriptions = {
            "target_sensor": f"目标列名，要预测的变量 (当前: {plsr_params['target_sensor']})",
            "n_components": f"PLS成分数，'auto'表示自动选择 (当前: {plsr_params['n_components']})",
            "scale": f"是否标准化数据 (当前: {plsr_params['scale']})",
            "max_iter": f"最大迭代次数 (当前: {plsr_params['max_iter']})",
            "tol": f"收敛阈值 (当前: {plsr_params['tol']})",
            "test_size": f"测试集比例 (当前: {plsr_params['test_size']})",
            "random_state": f"随机种子，确保结果可复现 (当前: {plsr_params['random_state']})",
            "cv_folds": f"交叉验证折数 (当前: {plsr_params['cv_folds']})",
            "max_components_to_try": f"最大尝试成分数 (当前: {plsr_params['max_components_to_try']})",
            "vip_threshold": f"VIP分数阈值 (当前: {plsr_params['vip_threshold']})"
        }

        for key, value in plsr_params.items():
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
        error_msg = f"PLSR模型运行失败: {e}"
        print(error_msg)
        result_str = "=======================\n"
        result_str += "PLSR模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 预测模型 PLSR 运行结束 <<<<<<<<<<<<<<<<<<<<")
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
            "vip_scores": {
                "data": vip_data_path if 'vip_data_path' in locals() else ""
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