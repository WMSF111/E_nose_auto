import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import json
import time
from datetime import datetime

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 分类模型 SVM 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"svm_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """SVM参数配置"""
        # 1. 扩展算法参数
        svm_params = {
            "kernel": "rbf",  # 默认径向基核
            "C": 1.0,  # 正则化参数
            "gamma": "scale",  # 核系数
            "degree": 3,  # 多项式核的阶数
            "coef0": 0.0,  # 核函数中的独立项
            "shrinking": True,  # 是否使用启发式收缩
            "probability": False,  # 是否启用概率估计
            "tol": 1e-3,  # 停止训练的误差值
            "cache_size": 200,  # 核函数缓存大小
            "random_state": 42
        }

        in_params = params["params"]

        if "kernel" in in_params:
            svm_params["kernel"] = in_params["kernel"]
        if "C" in in_params:
            svm_params["C"] = AlgModelParameters.format_to_float(in_params["C"])
        if "gamma" in in_params:
            if in_params["gamma"] in ["scale", "auto"]:
                svm_params["gamma"] = in_params["gamma"]
            else:
                svm_params["gamma"] = AlgModelParameters.format_to_float(in_params["gamma"])
        if "degree" in in_params:
            svm_params["degree"] = AlgModelParameters.format_to_int(in_params["degree"])
        if "coef0" in in_params:
            svm_params["coef0"] = AlgModelParameters.format_to_float(in_params["coef0"])
        if "shrinking" in in_params:
            svm_params["shrinking"] = in_params["shrinking"]
        if "probability" in in_params:
            svm_params["probability"] = in_params["probability"]
        if "tol" in in_params:
            svm_params["tol"] = AlgModelParameters.format_to_float(in_params["tol"])

        print(f"算法参数: {svm_params}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("分类模型", "SVM").copy()

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

        # 3. SVM模型训练
        svm = SVC(
            kernel=svm_params["kernel"],
            C=svm_params["C"],
            gamma=svm_params["gamma"],
            degree=svm_params["degree"],
            coef0=svm_params["coef0"],
            shrinking=svm_params["shrinking"],
            probability=svm_params["probability"],
            tol=svm_params["tol"],
            cache_size=svm_params["cache_size"],
            random_state=svm_params["random_state"]
        )
        svm.fit(X_train, y_train)

        # 4. 模型评估
        y_pred = svm.predict(X_test)
        y_train_pred = svm.predict(X_train)

        print("\n" + "=" * 50)
        print("SVM模型训练结果")
        print("=" * 50)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)

        print(f"\n训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        print("\n分类报告:")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report)

        # 5. 特征重要性（仅适用于线性核）
        if svm_params["kernel"] == "linear":
            feature_importance = np.abs(svm.coef_).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            # 保存特征重要性数据
            importance_path = os.path.join(output_dir, 'feature_importance.txt')
            importance_df.to_csv(importance_path, sep='\t', index=False)
        else:
            # 对于非线性SVM，无法直接获取特征重要性
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': np.zeros(len(X.columns))
            })
            importance_path = os.path.join(output_dir, 'feature_importance.txt')
            importance_df.to_csv(importance_path, sep='\t', index=False)

        # 6. 生成PCA降维散点图
        # 使用PCA进行降维可视化
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # 合并数据
        X_pca_all = np.vstack([X_train_pca, X_test_pca])
        y_all = np.concatenate([y_train, y_test])
        set_type = np.concatenate([['训练集'] * len(y_train), ['测试集'] * len(y_test)])

        # 保存数据
        scatter_data_path = os.path.join(output_dir, 'svm_pca_scatter_data.txt')
        scatter_data = pd.DataFrame({
            'PC1': X_pca_all[:, 0],
            'PC2': X_pca_all[:, 1] if X_pca_all.shape[1] > 1 else np.zeros(len(X_pca_all)),
            'class': [label_encoder.classes_[c] for c in y_all],
            'class_encoded': y_all,
            'set_type': set_type
        })
        scatter_data.to_csv(scatter_data_path, sep='\t', index=False)

        # 绘制PCA散点图
        plt.figure(figsize=(12, 8))

        # 绘制每个类别的数据点
        colors = plt.cm.Set2(np.linspace(0, 1, len(label_encoder.classes_)))

        for i, class_name in enumerate(label_encoder.classes_):
            # 训练集
            train_mask = (np.array([label_encoder.classes_[c] for c in y_all]) == class_name) & (set_type == '训练集')
            if np.any(train_mask):
                plt.scatter(X_pca_all[train_mask, 0], X_pca_all[train_mask, 1],
                            c=[colors[i]], alpha=0.6, s=60, marker='o',
                            label=f'{class_name} (训练集)', edgecolors='k', linewidth=0.5)

            # 测试集
            test_mask = (np.array([label_encoder.classes_[c] for c in y_all]) == class_name) & (set_type == '测试集')
            if np.any(test_mask):
                plt.scatter(X_pca_all[test_mask, 0], X_pca_all[test_mask, 1],
                            c=[colors[i]], alpha=0.8, s=80, marker='^',
                            label=f'{class_name} (测试集)', edgecolors='k', linewidth=1)

        # 标注支持向量（如果特征维度是2）
        if X_train.shape[1] <= 2:
            support_vectors = svm.support_vectors_
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                        s=100, facecolors='none', edgecolors='red',
                        linewidths=1.5, label='支持向量')

        plt.title(plot_config["scatter_title"], fontsize=16, fontweight='bold')
        plt.xlabel(f'{plot_config["scatter_xlabel"]} ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        if X_pca_all.shape[1] > 1:
            plt.ylabel(f'{plot_config["scatter_ylabel"]} ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        scatter_img_path = os.path.join(output_dir, 'svm_pca_scatter.png')
        plt.tight_layout()
        plt.savefig(scatter_img_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        imgs.append({
            "name": "SVM数据分布图（PCA降维）",
            "img": scatter_img_path,
            "data": scatter_data_path
        })

        # 7. 生成特征重要性条形图（仅线性核）
        if svm_params["kernel"] == "linear" and len(importance_df) > 0:
            plt.figure(figsize=(12, 6))
            top_n = min(15, len(importance_df))
            top_features = importance_df.head(top_n)

            plt.barh(range(len(top_features)), top_features['importance'][::-1])
            plt.yticks(range(len(top_features)), top_features['feature'][::-1])
            plt.xlabel(plot_config["importance_ylabel"], fontsize=12)
            plt.ylabel(plot_config["importance_xlabel"], fontsize=12)
            plt.title(plot_config["importance_title"], fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')

            importance_img_path = os.path.join(output_dir, 'feature_importance.png')
            plt.tight_layout()
            plt.savefig(importance_img_path, dpi=dpi)
            plt.close()

            imgs.append({
                "name": "特征重要性排名（线性核）",
                "img": importance_img_path,
                "data": importance_path
            })

        # 8. 生成混淆矩阵热图
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

        # 9. 生成PCA解释方差图（碎石图）
        if X_train.shape[1] > 2:
            pca_full = PCA()
            X_pca_full = pca_full.fit_transform(X_train)
            explained_variance_ratio = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            plt.figure(figsize=(12, 5))

            # 解释方差比例
            plt.subplot(1, 2, 1)
            components = np.arange(1, len(explained_variance_ratio) + 1)
            bars = plt.bar(components, explained_variance_ratio, color='steelblue', alpha=0.7)
            plt.xlabel(plot_config["scree_xlabel"], fontsize=12)
            plt.ylabel(plot_config["scree_ylabel"], fontsize=12)
            plt.title(plot_config["scree_title"], fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')

            # 在柱子上添加数值标签
            for bar, val in zip(bars, explained_variance_ratio):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.2%}', ha='center', va='bottom', fontsize=9)

            # 累计解释方差比例
            plt.subplot(1, 2, 2)
            plt.plot(components, cumulative_variance, 'o-', linewidth=2, markersize=8, color='darkorange')
            plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.6, label='95%阈值')
            plt.xlabel(plot_config["cumulative_xlabel"], fontsize=12)
            plt.ylabel(plot_config["cumulative_ylabel"], fontsize=12)
            plt.title(plot_config["cumulative_title"], fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 标记达到95%的组件数
            if cumulative_variance[-1] >= 0.95:
                idx = np.where(cumulative_variance >= 0.95)[0][0]
                plt.axvline(x=components[idx], color='g', linestyle=':', alpha=0.6)
                plt.text(components[idx], 0.5, f'主成分{components[idx]}\n达到95%',
                         ha='right', va='center', fontsize=9, color='g')

            scree_img_path = os.path.join(output_dir, 'pca_explained_variance.png')
            plt.tight_layout()
            plt.savefig(scree_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            scree_data_path = os.path.join(output_dir, 'pca_explained_variance_data.txt')
            scree_data = pd.DataFrame({
                'component': components,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance
            })
            scree_data.to_csv(scree_data_path, sep='\t', index=False)

            imgs.append({
                "name": "主成分解释方差图",
                "img": scree_img_path,
                "data": scree_data_path
            })

        # 10. 支持向量数量统计图
        plt.figure(figsize=(10, 6))
        support_counts = len(svm.support_)
        support_indices = svm.support_

        if X_train.shape[1] <= 2:
            # 2D可视化支持向量
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.5, s=30, label='所有样本')
            plt.scatter(X_train[support_indices, 0], X_train[support_indices, 1],
                        facecolors='none', edgecolors='black', s=100, linewidths=2, label='支持向量')
            plt.title(plot_config["support_vectors_title"], fontsize=16, fontweight='bold')
            plt.xlabel(plot_config["support_vectors_xlabel"], fontsize=12)
            plt.ylabel(plot_config["support_vectors_ylabel"], fontsize=12)
        else:
            # 仅显示支持向量数量统计
            plt.bar(['支持向量', '非支持向量'],
                    [support_counts, len(X_train) - support_counts],
                    color=['lightcoral', 'lightblue'])
            plt.title('支持向量统计', fontsize=16, fontweight='bold')
            plt.ylabel('数量', fontsize=12)
            plt.text(0, support_counts, str(support_counts), ha='center', va='bottom', fontsize=12)
            plt.text(1, len(X_train) - support_counts, str(len(X_train) - support_counts), ha='center', va='bottom',
                     fontsize=12)

        plt.legend()
        plt.grid(True, alpha=0.3)

        support_img_path = os.path.join(output_dir, 'support_vectors.png')
        plt.tight_layout()
        plt.savefig(support_img_path, dpi=dpi)
        plt.close()

        imgs.append({
            "name": "支持向量分布图",
            "img": support_img_path,
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
            "model_params": svm_params,
            "performance": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "support_vectors_count": int(support_counts),
                "support_vectors_ratio": float(support_counts / len(X_train))
            },
            "classification_report": classification_report(y_test, y_pred,
                                                           target_names=label_encoder.classes_,
                                                           output_dict=True),
            "top_features": importance_df.head(10).to_dict('records') if svm_params["kernel"] == "linear" else []
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 12. 生成结果字符串
        result_str += "=======================\n"
        result_str += "SVM模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {df.shape[1] - 1} 个特征\n"
        result_str += f"2. 类别数量: {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})\n"
        result_str += f"3. 核函数: {svm_params['kernel']}\n"
        result_str += f"4. 正则化参数C: {svm_params['C']}\n"
        result_str += f"5. 支持向量数量: {support_counts} ({support_counts / len(X_train):.2%})\n"
        result_str += f"6. 训练集准确率: {train_accuracy:.4f}\n"
        result_str += f"7. 测试集准确率: {test_accuracy:.4f}\n"

        # 判断是否过拟合
        if train_accuracy - test_accuracy > 0.1:
            result_str += "8. 模型可能存在过拟合\n"
        else:
            result_str += "8. 模型泛化能力良好\n"

        # 添加特征重要性（仅线性核）
        if svm_params["kernel"] == "linear" and len(importance_df) > 0:
            result_str += "\n=== 特征重要性排名 ===\n"
            result_str += str(importance_df.head(10)) + "\n"

        # 添加生成的图片信息
        result_str += f"\n=== 生成的图片 ===\n"
        for img_info in imgs:
            img_filename = os.path.basename(img_info['img'])
            data_filename = os.path.basename(img_info['data']) if img_info['data'] else "无数据文件"
            result_str += f"- {img_info['name']}: {img_filename} (数据文件: {data_filename})\n"

        # 添加特征重要性文件名
        result_str += f"\n=== 特征重要性文件 ===\n"
        importance_filename = os.path.basename(importance_path) if 'importance_path' in locals() else "未生成"
        result_str += f"特征重要性文件: {importance_filename}\n"

        # 添加模型信息文件名
        result_str += f"\n=== 模型信息文件 ===\n"
        model_info_filename = os.path.basename(model_info_path) if 'model_info_path' in locals() else "未生成"
        result_str += f"模型信息文件: {model_info_filename}\n"

        # 添加计算使用的参数信息
        result_str += f"\n=== 计算使用的参数信息 ===\n"
        for key, value in svm_params.items():
            result_str += f"  {key}: {value}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"SVM模型运行失败: {e}"
        print(error_msg)
        result_str = "=======================\n"
        result_str += "SVM模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 分类模型 SVM 运行结束 <<<<<<<<<<<<<<<<<<<<")
    print(f"计算耗时: {expend_time} 毫秒")

    result_str += f"\n计算耗时(毫秒):{expend_time}\n"

    # 构建JSON格式的结果
    result_json = {
        "success": True if "运行状态: 成功" in result_str else False,
        "error_msg": "" if "运行状态: 成功" in result_str else result_str.split("错误信息: ")[-1].strip(),
        "summary": result_str,
        "generated_files": {
            "imgs": imgs,  # 修改为imgs数组，与LDA一致
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