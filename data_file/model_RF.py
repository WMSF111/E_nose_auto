import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import os
import json
import time
from datetime import datetime

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 分类模型 RF 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"rf_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """RF参数配置"""
        # 1. 扩展算法参数
        rf_params = {
            "n_estimators": 100,  # 决策树数量
            "max_depth": None,  # 树的最大深度
            "min_samples_split": 2,  # 内部节点再划分所需最小样本数
            "min_samples_leaf": 1,  # 叶节点最少样本数
            "max_features": "sqrt",  # 寻找最佳分割时考虑的特征数量
            "bootstrap": True,  # 是否使用bootstrap采样
            "random_state": 42,  # 随机种子
            "oob_score": False,  # 是否使用袋外样本评估
            "max_samples": None,  # 从X抽取的样本数用于训练每个基学习器
            "ccp_alpha": 0.0,  # 最小代价复杂度剪枝
            "max_leaf_nodes": None  # 最大叶节点数
        }

        in_params = params["params"]

        if "n_estimators" in in_params:
            rf_params["n_estimators"] = AlgModelParameters.format_to_int(in_params["n_estimators"])
        if "max_depth" in in_params:
            rf_params["max_depth"] = AlgModelParameters.format_to_int(in_params["max_depth"])
        if "min_samples_split" in in_params:
            rf_params["min_samples_split"] = AlgModelParameters.format_to_int(in_params["min_samples_split"])
        if "min_samples_leaf" in in_params:
            rf_params["min_samples_leaf"] = AlgModelParameters.format_to_int(in_params["min_samples_leaf"])
        if "max_features" in in_params:
            rf_params["max_features"] = in_params["max_features"]
        if "bootstrap" in in_params:
            rf_params["bootstrap"] = in_params["bootstrap"]
        if "random_state" in in_params:
            rf_params["random_state"] = in_params["random_state"]
        if "oob_score" in in_params:
            rf_params["oob_score"] = in_params["oob_score"]
        if "ccp_alpha" in in_params:
            rf_params["ccp_alpha"] = AlgModelParameters.format_to_float(in_params["ccp_alpha"])

        print(f"算法参数: {rf_params}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("分类模型", "RF").copy()

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

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        print(f"\n训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")

        # 3. RF模型训练
        rf = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_features=rf_params["max_features"],
            bootstrap=rf_params["bootstrap"],
            random_state=rf_params["random_state"],
            oob_score=rf_params["oob_score"],
            ccp_alpha=rf_params["ccp_alpha"]
        )
        rf.fit(X_train, y_train)

        # 4. 模型评估
        y_pred = rf.predict(X_test)
        y_train_pred = rf.predict(X_train)

        print("\n" + "=" * 50)
        print("随机森林模型训练结果")
        print("=" * 50)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)

        print(f"\n训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        print("\n分类报告:")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report)

        # 5. 计算特征重要性
        feature_importance = rf.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        # 保存特征重要性数据
        importance_path = os.path.join(output_dir, 'feature_importance.txt')
        importance_df.to_csv(importance_path, sep='\t', index=False)

        # 6. 生成特征重要性条形图
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

        # 7. 生成混淆矩阵热图
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

        # 8. 生成每棵树的错误率分布图
        if rf_params["n_estimators"] <= 100:  # 树数量适中时才显示
            tree_errors = []
            for tree in rf.estimators_:
                tree_pred = tree.predict(X_test)
                tree_error = 1 - accuracy_score(y_test, tree_pred)
                tree_errors.append(tree_error)

            plt.figure(figsize=(10, 6))
            plt.hist(tree_errors, bins=20, edgecolor='black', alpha=0.7)
            plt.axvline(x=np.mean(tree_errors), color='red', linestyle='--',
                        linewidth=2, label=f'平均错误率: {np.mean(tree_errors):.4f}')
            plt.xlabel(plot_config["trees_error_xlabel"], fontsize=12)
            plt.ylabel(plot_config["trees_error_ylabel"], fontsize=12)
            plt.title(plot_config["trees_error_title"], fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            trees_error_img_path = os.path.join(output_dir, 'trees_error_distribution.png')
            plt.tight_layout()
            plt.savefig(trees_error_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            trees_error_data_path = os.path.join(output_dir, 'trees_error_distribution_data.txt')
            trees_error_data = pd.DataFrame({
                'tree_index': range(1, len(tree_errors) + 1),
                'error_rate': tree_errors
            })
            trees_error_data.to_csv(trees_error_data_path, sep='\t', index=False)

            imgs.append({
                "name": "每棵树错误率分布",
                "img": trees_error_img_path,
                "data": trees_error_data_path
            })

        # 9. 生成决策树深度分布图
        tree_depths = [tree.tree_.max_depth for tree in rf.estimators_]

        plt.figure(figsize=(10, 6))
        unique_depths = np.unique(tree_depths)
        depth_counts = [tree_depths.count(d) for d in unique_depths]

        plt.bar(unique_depths, depth_counts, edgecolor='black', alpha=0.7)
        plt.xlabel(plot_config["depth_dist_xlabel"], fontsize=12)
        plt.ylabel(plot_config["depth_dist_ylabel"], fontsize=12)
        plt.title(plot_config["depth_dist_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (depth, count) in enumerate(zip(unique_depths, depth_counts)):
            plt.text(depth, count + 0.1, str(count), ha='center', va='bottom', fontsize=10)

        depth_img_path = os.path.join(output_dir, 'tree_depth_distribution.png')
        plt.tight_layout()
        plt.savefig(depth_img_path, dpi=dpi)
        plt.close()

        # 保存数据
        depth_data_path = os.path.join(output_dir, 'tree_depth_distribution_data.txt')
        depth_data = pd.DataFrame({
            'tree_depth': tree_depths
        })
        depth_data['tree_depth'].value_counts().sort_index().to_csv(depth_data_path, sep='\t')

        imgs.append({
            "name": "决策树深度分布",
            "img": depth_img_path,
            "data": depth_data_path
        })

        # 10. 生成袋外误差图（如果启用了oob_score）
        if rf_params["oob_score"] and rf_params["bootstrap"]:
            # 训练不同树数量下的随机森林，计算OOB误差
            n_trees_range = np.arange(10, rf_params["n_estimators"] + 1,
                                      max(1, rf_params["n_estimators"] // 20))
            oob_errors = []

            for n in n_trees_range:
                rf_temp = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=rf_params["max_depth"],
                    min_samples_split=rf_params["min_samples_split"],
                    min_samples_leaf=rf_params["min_samples_leaf"],
                    max_features=rf_params["max_features"],
                    bootstrap=True,
                    oob_score=True,
                    random_state=rf_params["random_state"]
                )
                rf_temp.fit(X_train, y_train)
                oob_errors.append(1 - rf_temp.oob_score_)

            plt.figure(figsize=(10, 6))
            plt.plot(n_trees_range, oob_errors, 'o-', linewidth=2, markersize=6)
            plt.xlabel(plot_config["oob_error_xlabel"], fontsize=12)
            plt.ylabel(plot_config["oob_error_ylabel"], fontsize=12)
            plt.title(plot_config["oob_error_title"], fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)

            oob_img_path = os.path.join(output_dir, 'oob_error.png')
            plt.tight_layout()
            plt.savefig(oob_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            oob_data_path = os.path.join(output_dir, 'oob_error_data.txt')
            oob_data = pd.DataFrame({
                'n_estimators': n_trees_range,
                'oob_error': oob_errors
            })
            oob_data.to_csv(oob_data_path, sep='\t', index=False)

            imgs.append({
                "name": "袋外误差随树数量变化",
                "img": oob_img_path,
                "data": oob_data_path
            })

        # 11. 生成特征间相关性热图（对于特征数量适中的情况）
        if X.shape[1] <= 20:
            plt.figure(figsize=(12, 10))
            correlation_matrix = X.corr()

            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5},
                        annot=True if X.shape[1] <= 10 else False, fmt='.2f')
            plt.xlabel(plot_config["correlation_xlabel"], fontsize=12)
            plt.ylabel(plot_config["correlation_ylabel"], fontsize=12)
            plt.title(plot_config["correlation_title"], fontsize=16, fontweight='bold')

            correlation_img_path = os.path.join(output_dir, 'feature_correlation.png')
            plt.tight_layout()
            plt.savefig(correlation_img_path, dpi=dpi)
            plt.close()

            # 保存数据
            correlation_data_path = os.path.join(output_dir, 'feature_correlation_data.txt')
            correlation_matrix.to_csv(correlation_data_path, sep='\t')

            imgs.append({
                "name": "特征间相关性热图",
                "img": correlation_img_path,
                "data": correlation_data_path
            })

        # 12. 生成排列特征重要性图（更稳健的特征重要性评估）
        if X_test.shape[0] > 100:  # 有足够测试样本时才计算
            try:
                perm_importance = permutation_importance(rf, X_test, y_test,
                                                         n_repeats=10, random_state=42)

                perm_importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance_mean': perm_importance.importances_mean,
                    'importance_std': perm_importance.importances_std
                }).sort_values('importance_mean', ascending=False)

                plt.figure(figsize=(12, 6))
                top_n = min(15, len(perm_importance_df))
                top_perm_features = perm_importance_df.head(top_n)

                plt.barh(range(len(top_perm_features)),
                         top_perm_features['importance_mean'][::-1],
                         xerr=top_perm_features['importance_std'][::-1])
                plt.yticks(range(len(top_perm_features)),
                           top_perm_features['feature'][::-1])
                plt.xlabel(plot_config["perm_importance_ylabel"], fontsize=12)
                plt.ylabel(plot_config["perm_importance_xlabel"], fontsize=12)
                plt.title(plot_config["perm_importance_title"], fontsize=16, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='x')

                perm_img_path = os.path.join(output_dir, 'permutation_importance.png')
                plt.tight_layout()
                plt.savefig(perm_img_path, dpi=dpi)
                plt.close()

                # 保存数据
                perm_data_path = os.path.join(output_dir, 'permutation_importance_data.txt')
                perm_importance_df.to_csv(perm_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "排列特征重要性",
                    "img": perm_img_path,
                    "data": perm_data_path
                })
            except Exception as e:
                print(f"排列特征重要性计算失败: {e}")

        # 13. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": df.shape[0],
                "features": df.shape[1] - 1,
                "classes": len(label_encoder.classes_),
                "class_names": label_encoder.classes_.tolist()
            },
            "model_params": rf_params,
            "performance": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "n_trees": rf_params["n_estimators"],
                "avg_tree_depth": float(np.mean(tree_depths)),
                "oob_score": float(rf.oob_score_) if rf_params["oob_score"] else None
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

        # 14. 生成结果字符串
        result_str += "=======================\n"
        result_str += "随机森林模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {df.shape[1] - 1} 个特征\n"
        result_str += f"2. 类别数量: {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})\n"
        result_str += f"3. 决策树数量: {rf_params['n_estimators']}\n"
        result_str += f"4. 平均树深度: {np.mean(tree_depths):.2f}\n"
        result_str += f"5. 训练集准确率: {train_accuracy:.4f}\n"
        result_str += f"6. 测试集准确率: {test_accuracy:.4f}\n"

        if rf_params["oob_score"] and rf_params["bootstrap"]:
            result_str += f"7. 袋外得分: {rf.oob_score_:.4f}\n"

        # 判断是否过拟合
        if train_accuracy - test_accuracy > 0.1:
            result_str += "8. 模型可能存在过拟合\n"
        else:
            result_str += "8. 模型泛化能力良好\n"

        # 添加特征重要性
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
        for key, value in rf_params.items():
            result_str += f"  {key}: {value}\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"随机森林模型运行失败: {e}"
        print(error_msg)
        result_str = "=======================\n"
        result_str += "随机森林模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 分类模型 RF 运行结束 <<<<<<<<<<<<<<<<<<<<")
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