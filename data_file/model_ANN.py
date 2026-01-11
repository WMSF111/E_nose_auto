import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import json
import time
from datetime import datetime

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 分类模型 ANN 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"ann_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """ANN参数配置 - 带详细注释"""
        # 1. 扩展算法参数
        ann_params = {
            "hidden_layer_sizes": (100, 50),
            # 隐藏层结构
            # 含义：指定神经网络的隐藏层结构和每层的神经元数量
            # 格式：元组，例如(100, 50)表示2个隐藏层，第一层100个神经元，第二层50个神经元
            # 取值范围：正整数元组，例如(50,), (100, 50), (200, 100, 50)
            # 默认值：(100,)（单隐藏层100个神经元）

            "activation": "relu",
            # 激活函数
            # 含义：隐藏层的激活函数类型
            # 可选值：
            # - 'identity': 线性激活函数 f(x) = x
            # - 'logistic': Logistic sigmoid函数，f(x) = 1 / (1 + exp(-x))
            # - 'tanh': 双曲正切函数，f(x) = tanh(x)
            # - 'relu': 修正线性单元，f(x) = max(0, x)（默认）
            # 取值范围：上述四个字符串之一

            "solver": "adam",
            # 优化算法
            # 含义：用于权重优化的求解器
            # 可选值：
            # - 'lbfgs': 准牛顿方法，适合小数据集
            # - 'sgd': 随机梯度下降
            # - 'adam': 基于随机梯度的优化器，适合大数据集（默认）
            # 取值范围：上述三个字符串之一

            "alpha": 0.0001,
            # L2正则化参数
            # 含义：控制模型复杂度的正则化项，防止过拟合
            # 取值范围：正浮点数，通常10^(-5)到10^(-1)
            # 默认值：0.0001

            "batch_size": "auto",
            # 批大小
            # 含义：每次梯度更新中使用的样本数
            # 可选值：
            # - 'auto': batch_size = min(200, n_samples)（默认）
            # - 正整数：指定批量大小
            # 取值范围：正整数或'auto'

            "learning_rate": "constant",
            # 学习率策略
            # 含义：学习率的更新策略
            # 可选值：
            # - 'constant': 恒定学习率（默认）
            # - 'invscaling': 随时间逐渐减小，learning_rate_init / t^power_t
            # - 'adaptive': 当训练损失不再下降时减小学习率
            # 取值范围：上述三个字符串之一

            "learning_rate_init": 0.001,
            # 初始学习率
            # 含义：优化器的初始学习率
            # 取值范围：正浮点数，通常0.0001到0.1
            # 默认值：0.001

            "max_iter": 200,
            # 最大迭代次数
            # 含义：优化器的最大迭代次数
            # 取值范围：正整数，通常100-1000
            # 默认值：200

            "shuffle": True,
            # 是否打乱数据
            # 含义：是否在每个epoch打乱训练样本
            # 取值范围：布尔值True或False
            # 默认值：True

            "random_state": 42,
            # 随机种子
            # 含义：控制随机数生成，确保结果可复现
            # 取值范围：正整数或None
            # 默认值：None（随机结果）

            "tol": 1e-4,
            # 训练停止阈值
            # 含义：当训练损失改进小于该值时停止训练
            # 取值范围：正浮点数，通常1e-5到1e-3
            # 默认值：1e-4

            "verbose": False,
            # 是否显示训练过程
            # 含义：是否打印训练进度信息
            # 取值范围：布尔值True或False
            # 默认值：False

            "warm_start": False,
            # 是否热启动
            # 含义：是否重用前一次调用的解作为初始化
            # 取值范围：布尔值True或False
            # 默认值：False

            "momentum": 0.9,
            # 动量参数（仅对sgd有效）
            # 含义：梯度下降的动量因子，加速收敛
            # 取值范围：0到1之间的浮点数
            # 默认值：0.9

            "nesterovs_momentum": True,
            # 是否使用Nesterov动量（仅对sgd有效）
            # 含义：是否使用Nesterov动量加速
            # 取值范围：布尔值True或False
            # 默认值：True

            "early_stopping": False,
            # 是否使用早停
            # 含义：当验证分数不再提高时提前停止训练
            # 取值范围：布尔值True或False
            # 默认值：False

            "validation_fraction": 0.1,
            # 验证集比例（早停时使用）
            # 含义：用于早停的验证集比例
            # 取值范围：0到1之间的浮点数
            # 默认值：0.1

            "beta_1": 0.9,
            # Adam的beta1参数
            # 含义：Adam优化器的一阶矩估计衰减率
            # 取值范围：0到1之间的浮点数
            # 默认值：0.9

            "beta_2": 0.999,
            # Adam的beta2参数
            # 含义：Adam优化器的二阶矩估计衰减率
            # 取值范围：0到1之间的浮点数
            # 默认值：0.999

            "epsilon": 1e-8,
            # Adam的epsilon参数
            # 含义：Adam优化器的数值稳定常数
            # 取值范围：正浮点数，通常1e-8到1e-7
            # 默认值：1e-8

            "n_iter_no_change": 10,
            # 早停等待轮数
            # 含义：在早停之前等待的epoch数
            # 取值范围：正整数，通常5-20
            # 默认值：10

            "max_fun": 15000
            # 最大函数评估次数（仅对lbfgs有效）
            # 含义：lbfgs求解器的最大函数评估次数
            # 取值范围：正整数
            # 默认值：15000
        }

        # 从传入参数中更新ANN参数
        in_params = params["params"]

        # 解析隐藏层结构
        if "hidden_layer_sizes" in in_params:
            try:
                # 尝试将字符串转换为元组，例如 "(100, 50)" -> (100, 50)
                hidden_str = in_params["hidden_layer_sizes"]
                hidden_str = hidden_str.strip("()")
                hidden_tuple = tuple(map(int, hidden_str.split(",")))
                ann_params["hidden_layer_sizes"] = hidden_tuple
            except:
                # 如果转换失败，使用默认值
                print(f"警告：无法解析隐藏层结构 '{in_params['hidden_layer_sizes']}'，使用默认值")

        if "activation" in in_params:
            ann_params["activation"] = in_params["activation"]
        if "solver" in in_params:
            ann_params["solver"] = in_params["solver"]
        if "alpha" in in_params:
            ann_params["alpha"] = AlgModelParameters.format_to_float(in_params["alpha"])
        if "learning_rate" in in_params:
            ann_params["learning_rate"] = in_params["learning_rate"]
        if "learning_rate_init" in in_params:
            ann_params["learning_rate_init"] = AlgModelParameters.format_to_float(in_params["learning_rate_init"])
        if "max_iter" in in_params:
            ann_params["max_iter"] = AlgModelParameters.format_to_int(in_params["max_iter"])
        if "random_state" in in_params:
            ann_params["random_state"] = AlgModelParameters.format_to_int(in_params["random_state"])
        if "tol" in in_params:
            ann_params["tol"] = AlgModelParameters.format_to_float(in_params["tol"])
        if "momentum" in in_params:
            ann_params["momentum"] = AlgModelParameters.format_to_float(in_params["momentum"])
        if "beta_1" in in_params:
            ann_params["beta_1"] = AlgModelParameters.format_to_float(in_params["beta_1"])
        if "beta_2" in in_params:
            ann_params["beta_2"] = AlgModelParameters.format_to_float(in_params["beta_2"])
        if "epsilon" in in_params:
            ann_params["epsilon"] = AlgModelParameters.format_to_float(in_params["epsilon"])

        print(f"算法参数: {ann_params}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("分类模型", "ANN").copy()

        # 如果用户提供了自定义图片参数，则更新
        if "image_param" in params and isinstance(params["image_param"], dict):
            plot_config.update(params["image_param"])

        # 提取特征和标签
        X = df.drop('target', axis=1)
        y = df['target']

        # 编码标签
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"\n编码后的标签: {label_encoder.classes_}")

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        print(f"\n训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")

        # 3. ANN模型训练
        # 启用warm_start和early_stopping以获取训练历史
        ann_params_with_history = ann_params.copy()
        ann_params_with_history["warm_start"] = True
        ann_params_with_history["early_stopping"] = True

        # 创建并训练ANN模型
        ann = MLPClassifier(
            hidden_layer_sizes=ann_params["hidden_layer_sizes"],
            activation=ann_params["activation"],
            solver=ann_params["solver"],
            alpha=ann_params["alpha"],
            batch_size=ann_params["batch_size"],
            learning_rate=ann_params["learning_rate"],
            learning_rate_init=ann_params["learning_rate_init"],
            max_iter=ann_params["max_iter"],
            shuffle=ann_params["shuffle"],
            random_state=ann_params["random_state"],
            tol=ann_params["tol"],
            verbose=False,
            warm_start=False,
            momentum=ann_params["momentum"],
            nesterovs_momentum=ann_params["nesterovs_momentum"],
            early_stopping=ann_params["early_stopping"],
            validation_fraction=ann_params["validation_fraction"],
            beta_1=ann_params["beta_1"],
            beta_2=ann_params["beta_2"],
            epsilon=ann_params["epsilon"],
            n_iter_no_change=ann_params["n_iter_no_change"],
            max_fun=ann_params["max_fun"]
        )

        print("\n训练ANN模型...")
        ann.fit(X_train, y_train)

        # 4. 模型评估
        y_pred = ann.predict(X_test)
        y_train_pred = ann.predict(X_train)

        print("\n" + "=" * 50)
        print("ANN模型训练结果")
        print("=" * 50)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)

        print(f"\n训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        print("\n分类报告:")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report)

        # 5. 生成训练损失曲线图（如果可用）
        if hasattr(ann, 'loss_curve_'):
            loss_curve = ann.loss_curve_

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(loss_curve) + 1), loss_curve, 'b-', linewidth=2)
            plt.xlabel(plot_config["loss_curve_xlabel"], fontsize=12)
            plt.ylabel(plot_config["loss_curve_ylabel"], fontsize=12)
            plt.title(plot_config["loss_curve_title"], fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # 标记最终损失值
            final_loss = loss_curve[-1]
            plt.axhline(y=final_loss, color='r', linestyle='--', alpha=0.7)
            plt.text(len(loss_curve), final_loss, f' 最终损失: {final_loss:.4f}',
                     verticalalignment='bottom', fontsize=11)

            loss_curve_img_path = os.path.join(output_dir, 'loss_curve.png')
            plt.tight_layout()
            plt.savefig(loss_curve_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            loss_curve_data_path = os.path.join(output_dir, 'loss_curve_data.txt')
            loss_curve_data = pd.DataFrame({
                'epoch': range(1, len(loss_curve) + 1),
                'loss': loss_curve
            })
            loss_curve_data.to_csv(loss_curve_data_path, sep='\t', index=False)

            imgs.append({
                "name": "训练损失曲线",
                "img": loss_curve_img_path,
                "data": loss_curve_data_path
            })

        # 6. 生成混淆矩阵热图
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.xlabel(plot_config["confusion_xlabel"], fontsize=12)
        plt.ylabel(plot_config["confusion_ylabel"], fontsize=12)
        plt.title(plot_config["confusion_title"], fontsize=16, fontweight='bold')

        confusion_img_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(confusion_img_path, dpi=dpi)
        plt.close()

        # 保存混淆矩阵数据
        confusion_data_path = os.path.join(output_dir, 'confusion_matrix_data.txt')
        cm_df = pd.DataFrame(cm,
                             index=[f"True_{c}" for c in label_encoder.classes_],
                             columns=[f"Pred_{c}" for c in label_encoder.classes_])
        cm_df.to_csv(confusion_data_path, sep='\t')

        imgs.append({
            "name": "混淆矩阵热图",
            "img": confusion_img_path,
            "data": confusion_data_path
        })

        # 7. 生成基于权重的特征重要性图
        if hasattr(ann, 'coefs_') and len(ann.coefs_) > 0:
            # 使用第一层权重计算特征重要性
            first_layer_weights = np.abs(ann.coefs_[0])
            feature_importance = np.sum(first_layer_weights, axis=1)

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            # 保存特征重要性数据
            importance_path = os.path.join(output_dir, 'feature_importance.txt')
            importance_df.to_csv(importance_path, sep='\t', index=False)

            # 绘制特征重要性条形图
            plt.figure(figsize=(12, 6))
            top_n = min(15, len(importance_df))
            top_features = importance_df.head(top_n)

            plt.barh(range(len(top_features)), top_features['importance'][::-1])
            plt.yticks(range(len(top_features)), top_features['feature'][::-1])
            plt.xlabel(plot_config["feature_importance_ylabel"], fontsize=12)
            plt.ylabel(plot_config["feature_importance_xlabel"], fontsize=12)
            plt.title(plot_config["feature_importance_title"], fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')

            importance_img_path = os.path.join(output_dir, 'feature_importance.png')
            plt.tight_layout()
            plt.savefig(importance_img_path, dpi=dpi)
            plt.close()

            imgs.append({
                "name": "基于权重的特征重要性",
                "img": importance_img_path,
                "data": importance_path
            })

        # 8. 生成神经网络权重分布图
        if hasattr(ann, 'coefs_'):
            plt.figure(figsize=(12, 6))

            all_weights = []
            for i, coef in enumerate(ann.coefs_):
                all_weights.extend(coef.flatten())

            plt.hist(all_weights, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel(plot_config["layer_weights_xlabel"], fontsize=12)
            plt.ylabel(plot_config["layer_weights_ylabel"], fontsize=12)
            plt.title(plot_config["layer_weights_title"], fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # 添加统计信息
            mean_weight = np.mean(all_weights)
            std_weight = np.std(all_weights)
            plt.axvline(x=mean_weight, color='red', linestyle='--', linewidth=2,
                        label=f'均值: {mean_weight:.4f}')
            plt.legend()

            weights_img_path = os.path.join(output_dir, 'weights_distribution.png')
            plt.tight_layout()
            plt.savefig(weights_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            weights_data_path = os.path.join(output_dir, 'weights_distribution_data.txt')
            weights_data = pd.DataFrame({
                'weight_value': all_weights
            })
            weights_data.to_csv(weights_data_path, sep='\t', index=False)

            imgs.append({
                "name": "神经网络权重分布",
                "img": weights_img_path,
                "data": weights_data_path
            })

        # 9. 生成预测概率分布图
        if hasattr(ann, 'predict_proba'):
            y_proba = ann.predict_proba(X_test)

            plt.figure(figsize=(12, 6))

            # 对于每个类别，绘制预测概率的分布
            for i in range(y_proba.shape[1]):
                plt.hist(y_proba[:, i], bins=30, alpha=0.5, label=f'类别 {label_encoder.classes_[i]}')

            plt.xlabel(plot_config["class_probability_xlabel"], fontsize=12)
            plt.ylabel(plot_config["class_probability_ylabel"], fontsize=12)
            plt.title(plot_config["class_probability_title"], fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            probability_img_path = os.path.join(output_dir, 'probability_distribution.png')
            plt.tight_layout()
            plt.savefig(probability_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            probability_data_path = os.path.join(output_dir, 'probability_distribution_data.txt')
            probability_data = pd.DataFrame(y_proba, columns=[f'prob_class_{c}' for c in label_encoder.classes_])
            probability_data['true_class'] = [label_encoder.classes_[c] for c in y_test]
            probability_data.to_csv(probability_data_path, sep='\t', index=False)

            imgs.append({
                "name": "类别概率分布",
                "img": probability_img_path,
                "data": probability_data_path
            })

        # 10. 生成网络结构信息图
        plt.figure(figsize=(10, 6))

        layer_sizes = [X.shape[1]] + list(ann_params["hidden_layer_sizes"]) + [len(label_encoder.classes_)]

        # 绘制简单的网络结构示意图
        for layer_idx, size in enumerate(layer_sizes):
            x_pos = layer_idx
            y_positions = np.linspace(-size / 2, size / 2, size)

            # 绘制神经元
            plt.scatter([x_pos] * size, y_positions, s=200,
                        c='lightblue' if layer_idx == 0 else (
                            'lightgreen' if layer_idx < len(layer_sizes) - 1 else 'lightcoral'),
                        edgecolors='black', zorder=2)

            # 标注神经元数量
            plt.text(x_pos, max(y_positions) + 1, f'{size}', ha='center', fontsize=10, fontweight='bold')

            # 绘制层标签
            if layer_idx == 0:
                layer_name = "输入层"
            elif layer_idx == len(layer_sizes) - 1:
                layer_name = "输出层"
            else:
                layer_name = f"隐藏层{layer_idx}"

            plt.text(x_pos, min(y_positions) - 2, layer_name, ha='center', fontsize=11, fontweight='bold')

        plt.xlim(-0.5, len(layer_sizes) - 0.5)
        plt.ylim(-max(layer_sizes) / 2 - 3, max(layer_sizes) / 2 + 3)
        plt.axis('off')
        plt.title(f'神经网络结构图\n总参数: {ann.n_outputs_ * ann.n_features_in_:,}', fontsize=14, fontweight='bold')

        structure_img_path = os.path.join(output_dir, 'network_structure.png')
        plt.tight_layout()
        plt.savefig(structure_img_path, dpi=dpi)
        plt.close()

        imgs.append({
            "name": "神经网络结构图",
            "img": structure_img_path,
            "data": ""
        })

        # 11. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": df.shape[0],
                "features": df.shape[1] - 1,
                "classes": len(label_encoder.classes_),
                "class_names": label_encoder.classes_.tolist()
            },
            "model_params": ann_params,
            "performance": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "final_loss": float(ann.loss_) if hasattr(ann, 'loss_') else None,
                "n_iterations": int(ann.n_iter_) if hasattr(ann, 'n_iter_') else None,
                "n_layers": len(ann.coefs_) if hasattr(ann, 'coefs_') else None
            },
            "classification_report": classification_report(y_test, y_pred,
                                                           target_names=label_encoder.classes_,
                                                           output_dict=True),
            "network_architecture": {
                "input_size": int(X.shape[1]),
                "hidden_layers": list(ann_params["hidden_layer_sizes"]),
                "output_size": int(len(label_encoder.classes_))
            }
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 12. 生成结果字符串
        result_str += "=======================\n"
        result_str += "ANN模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {df.shape[1] - 1} 个特征\n"
        result_str += f"2. 类别数量: {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})\n"
        result_str += f"3. 网络结构: 输入层({X.shape[1]}) -> "
        for i, size in enumerate(ann_params["hidden_layer_sizes"]):
            result_str += f"隐藏层{i + 1}({size}) -> "
        result_str += f"输出层({len(label_encoder.classes_)})\n"
        result_str += f"4. 激活函数: {ann_params['activation']}\n"
        result_str += f"5. 优化算法: {ann_params['solver']}\n"
        result_str += f"6. 训练集准确率: {train_accuracy:.4f}\n"
        result_str += f"7. 测试集准确率: {test_accuracy:.4f}\n"

        if hasattr(ann, 'loss_'):
            result_str += f"8. 最终损失值: {ann.loss_:.4f}\n"

        if hasattr(ann, 'n_iter_'):
            result_str += f"9. 训练迭代次数: {ann.n_iter_}\n"

        # 判断是否过拟合
        if train_accuracy - test_accuracy > 0.1:
            result_str += "10. 模型可能存在过拟合\n"
        else:
            result_str += "10. 模型泛化能力良好\n"

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
        result_str += f"ANN算法参数:\n"

        # 参数详细说明
        param_descriptions = {
            "hidden_layer_sizes": "隐藏层结构，元组形式，如(100, 50)表示两个隐藏层",
            "activation": f"激活函数，可选值: 'identity', 'logistic', 'tanh', 'relu' (当前: {ann_params['activation']})",
            "solver": f"优化算法，可选值: 'lbfgs', 'sgd', 'adam' (当前: {ann_params['solver']})",
            "alpha": f"L2正则化参数，防止过拟合 (当前: {ann_params['alpha']})",
            "learning_rate": f"学习率策略，可选值: 'constant', 'invscaling', 'adaptive' (当前: {ann_params['learning_rate']})",
            "learning_rate_init": f"初始学习率 (当前: {ann_params['learning_rate_init']})",
            "max_iter": f"最大迭代次数 (当前: {ann_params['max_iter']})",
            "random_state": f"随机种子，确保结果可复现 (当前: {ann_params['random_state']})",
            "tol": f"训练停止阈值 (当前: {ann_params['tol']})",
            "momentum": f"动量参数，加速梯度下降 (当前: {ann_params['momentum']})",
            "beta_1": f"Adam优化器的beta1参数 (当前: {ann_params['beta_1']})",
            "beta_2": f"Adam优化器的beta2参数 (当前: {ann_params['beta_2']})"
        }

        for key, value in ann_params.items():
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
        error_msg = f"ANN模型运行失败: {e}"
        print(error_msg)
        result_str = "=======================\n"
        result_str += "ANN模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 分类模型 ANN 运行结束 <<<<<<<<<<<<<<<<<<<<")
    print(f"计算耗时: {expend_time} 毫秒")

    result_str += f"\n计算耗时(毫秒):{expend_time}\n"

    # 构建JSON格式的结果
    result_json = {
        "success": True if "运行状态: 成功" in result_str else False,
        "error_msg": "" if "运行状态: 成功" in result_str else result_str.split("错误信息: ")[-1].strip(),
        "summary": result_str,
        "generated_files": {
            "imgs": imgs,  # 与LDA一致的结构
            "model_info": {
                "data": model_info_path if 'model_info_path' in locals() else ""
            },
            "feature_importance": {
                "data": importance_path if 'importance_path' in locals() else ""
            }
        },
        "start": start_timestamp,
        "end": end_timestamp,
        "expend": expend_time
    }

    # 将结果转换为JSON字符串返回
    return json.dumps(result_json, ensure_ascii=False, indent=2)