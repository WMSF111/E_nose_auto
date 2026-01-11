import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import json
import time
from datetime import datetime

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 分类模型 LDA 运行 >>>>>>>>>>>>>>>>>>>>")

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
            "solver": "svd",  # 默认值
            "shrinkage": None,
            "priors": None,  # 类别先验概率
            "n_components": None,
            "store_covariance": False,
            "tol": 1e-4
        }

        in_params = params["params"]

        if "solver" in in_params:
            lda_params["solver"] = in_params["solver"]
        if "shrinkage" in in_params:
            lda_params["shrinkage"] = AlgModelParameters.format_to_float(in_params["shrinkage"])
        if "n_components" in in_params:
            lda_params["n_components"] = AlgModelParameters.format_to_int(in_params["n_components"])
        if "store_covariance" in in_params:
            lda_params["store_covariance"] = in_params["store_covariance"]

        print(f"算法参数: {lda_params}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("分类模型", "LDA").copy()

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

        # 3. LDA模型训练
        lda = LinearDiscriminantAnalysis(
            solver=lda_params["solver"],
            shrinkage=lda_params["shrinkage"],
            priors=lda_params["priors"],
            n_components=lda_params["n_components"],
            store_covariance=lda_params["store_covariance"],
            tol=lda_params["tol"]
        )
        lda.fit(X_train, y_train)

        # 4. 模型评估
        y_pred = lda.predict(X_test)
        y_train_pred = lda.predict(X_train)

        print("\n" + "=" * 50)
        print("LDA模型训练结果")
        print("=" * 50)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)

        print(f"\n训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        print("\n分类报告:")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report)

        # 5. 计算特征重要性
        feature_importance = np.abs(lda.coef_).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        # 保存特征重要性数据
        importance_path = os.path.join(output_dir, 'feature_importance.txt')
        importance_df.to_csv(importance_path, sep='\t', index=False)

        # 6. 生成判别轴投影图（散点图）
        if X_train.shape[1] >= 2 and len(label_encoder.classes_) >= 2:
            # 获取LDA转换后的数据
            X_train_lda = lda.transform(X_train)
            X_test_lda = lda.transform(X_test)

            # 合并数据
            X_lda_all = np.vstack([X_train_lda, X_test_lda])
            y_all = np.concatenate([y_train, y_test])
            set_type = np.concatenate([['训练集'] * len(y_train), ['测试集'] * len(y_test)])

            # 保存数据
            scatter_data_path = os.path.join(output_dir, 'lda_projection_data.txt')
            scatter_data = pd.DataFrame({
                'LD1': X_lda_all[:, 0],
                'LD2': X_lda_all[:, 1] if X_lda_all.shape[1] > 1 else np.zeros(len(X_lda_all)),
                'class': [label_encoder.classes_[c] for c in y_all],
                'class_encoded': y_all,
                'set_type': set_type
            })
            scatter_data.to_csv(scatter_data_path, sep='\t', index=False)

            # 绘制判别轴投影图
            plt.figure(figsize=(12, 8))

            if X_lda_all.shape[1] >= 2:
                # 绘制每个类别的数据点
                colors = plt.cm.Set2(np.linspace(0, 1, len(label_encoder.classes_)))

                for i, class_name in enumerate(label_encoder.classes_):
                    # 训练集
                    train_mask = (np.array([label_encoder.classes_[c] for c in y_all]) == class_name) & (
                                set_type == '训练集')
                    if np.any(train_mask):
                        plt.scatter(X_lda_all[train_mask, 0], X_lda_all[train_mask, 1],
                                    c=[colors[i]], alpha=0.6, s=60, marker='o',
                                    label=f'{class_name} (训练集)', edgecolors='k', linewidth=0.5)

                    # 测试集
                    test_mask = (np.array([label_encoder.classes_[c] for c in y_all]) == class_name) & (
                                set_type == '测试集')
                    if np.any(test_mask):
                        plt.scatter(X_lda_all[test_mask, 0], X_lda_all[test_mask, 1],
                                    c=[colors[i]], alpha=0.8, s=80, marker='^',
                                    label=f'{class_name} (测试集)', edgecolors='k', linewidth=1)

                plt.title(plot_config["scatter_title"], fontsize=16, fontweight='bold')
                plt.xlabel(plot_config["scatter_xlabel"], fontsize=12)
                plt.ylabel(plot_config["scatter_ylabel"], fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)

                scatter_img_path = os.path.join(output_dir, 'lda_projection.png')
                plt.tight_layout()
                plt.savefig(scatter_img_path, dpi=dpi, bbox_inches='tight')
                plt.close()

                imgs.append({
                    "name": "LDA判别轴投影图",
                    "img": scatter_img_path,
                    "data": scatter_data_path
                })

        # 7. 生成特征重要性条形图
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
            "name": "特征重要性排名",
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

        # 9. 生成判别轴解释方差图（碎石图）
        if hasattr(lda, 'explained_variance_ratio_'):
            explained_variance_ratio = lda.explained_variance_ratio_
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
                plt.text(components[idx], 0.5, f'组件{components[idx]}\n达到95%',
                         ha='right', va='center', fontsize=9, color='g')

            scree_img_path = os.path.join(output_dir, 'explained_variance.png')
            plt.tight_layout()
            plt.savefig(scree_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            scree_data_path = os.path.join(output_dir, 'explained_variance_data.txt')
            scree_data = pd.DataFrame({
                'component': components,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance
            })
            scree_data.to_csv(scree_data_path, sep='\t', index=False)

            imgs.append({
                "name": "判别轴解释方差图",
                "img": scree_img_path,
                "data": scree_data_path
            })

        # 10. 生成判别系数热图（对于特征多的情况）
        if X.shape[1] <= 20:  # 特征数量适中时才显示热图
            plt.figure(figsize=(12, 6))
            coef_df = pd.DataFrame(lda.coef_,
                                   index=[f'判别函数{i + 1}' for i in range(lda.coef_.shape[0])],
                                   columns=X.columns)

            sns.heatmap(coef_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                        linewidths=0.5, linecolor='gray')
            plt.xlabel(plot_config["coef_heatmap_xlabel"], fontsize=12)
            plt.ylabel(plot_config["coef_heatmap_ylabel"], fontsize=12)
            plt.title(plot_config["coef_heatmap_title"], fontsize=14, fontweight='bold')

            coef_img_path = os.path.join(output_dir, 'discriminant_coefficients.png')
            plt.tight_layout()
            plt.savefig(coef_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            coef_data_path = os.path.join(output_dir, 'discriminant_coefficients_data.txt')
            coef_df.to_csv(coef_data_path, sep='\t')

            imgs.append({
                "name": "判别系数热图",
                "img": coef_img_path,
                "data": coef_data_path
            })

        # 11. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": df.shape[0],
                "features": df.shape[1] - 1,
                "classes": len(label_encoder.classes_),
                "class_names": label_encoder.classes_.tolist()
            },
            "model_params": lda_params,
            "performance": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "lda_components": int(lda.coef_.shape[0]),
                "n_features": X.shape[1]
            },
            "classification_report": classification_report(y_test, y_pred,
                                                           target_names=label_encoder.classes_,
                                                           output_dict=True),
            "top_features": importance_df.head(10).to_dict('records')
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 12. 生成结果字符串
        result_str += "=======================\n"
        result_str += "LDA模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {df.shape[1] - 1} 个特征\n"
        result_str += f"2. 类别数量: {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})\n"
        result_str += f"3. LDA判别函数数量: {lda.coef_.shape[0]}\n"
        result_str += f"4. 训练集准确率: {train_accuracy:.4f}\n"
        result_str += f"5. 测试集准确率: {test_accuracy:.4f}\n"

        if hasattr(lda, 'explained_variance_ratio_'):
            result_str += f"6. 累计解释方差: {np.sum(lda.explained_variance_ratio_):.4f}\n"

        # 判断是否过拟合
        if train_accuracy - test_accuracy > 0.1:
            result_str += "7. 模型可能存在过拟合\n"
        else:
            result_str += "7. 模型泛化能力良好\n"

        # 添加特征重要性
        result_str += "\n=== 特征重要性排名 ===\n"
        result_str += str(importance_df.head(10)) + "\n"

        # 添加生成的图片信息（只显示文件名）
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
        for key, value in lda_params.items():
            result_str += f"  {key}: {value}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"LDA模型运行失败: {e}"
        print(error_msg)
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

    print("<<<<<<<<<<<<<<<<<<<< 分类模型 LDA 运行结束 <<<<<<<<<<<<<<<<<<<<")
    print(f"计算耗时: {expend_time} 毫秒")

    result_str += f"\n计算耗时(毫秒):{expend_time}\n"

    # 构建JSON格式的结果
    result_json = {
        "success": True if "运行状态: 成功" in result_str else False,
        "error_msg": "" if "运行状态: 成功" in result_str else result_str.split("错误信息: ")[-1].strip(),
        "summary": result_str,
        "generated_files": {
            "imgs": imgs,  # 修改为imgs数组
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