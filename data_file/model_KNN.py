import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
import os
import json
import time
from datetime import datetime

from tool.UI_show.alg import AlgModelParameters

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

warnings.filterwarnings('ignore')


def run(df, dir, params, dpi):
    print(">>>>>>>>>>>>>>>>>>>> 分类模型 KNN 运行 >>>>>>>>>>>>>>>>>>>>")

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_millis = int(time.time() * 1000)

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(dir, f"knn_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字符串
    result_str = ""

    # 初始化图片列表
    imgs = []

    try:
        """KNN参数配置"""
        # 1. 扩展算法参数（带详细注释）
        knn_params = {
            "n_neighbors": 5,
            # K值，最近邻的数量
            # 可选值范围：正整数，通常1-20之间，默认5
            # 太小可能导致过拟合，太大可能导致欠拟合

            "weights": "uniform",
            # 权重类型
            # 可选值："uniform"（均匀权重），"distance"（距离反比权重）
            # "uniform"：所有近邻的权重相等
            # "distance"：权重与距离成反比，距离越近权重越大

            "algorithm": "auto",
            # 计算最近邻的算法
            # 可选值："auto", "ball_tree", "kd_tree", "brute"
            # "auto"：根据数据自动选择最合适的算法
            # "ball_tree"：适用于高维数据
            # "kd_tree"：适用于低维数据（维度<20）
            # "brute"：暴力搜索，适用于小数据集

            "leaf_size": 30,
            # 叶节点大小（仅对ball_tree和kd_tree有效）
            # 可选值范围：正整数，默认30
            # 影响树构建的速度和内存使用

            "p": 2,
            # 距离度量参数
            # 可选值：1（曼哈顿距离），2（欧氏距离），或其他正整数
            # p=1：曼哈顿距离（L1距离）
            # p=2：欧氏距离（L2距离）
            # p>2：明可夫斯基距离

            "metric": "minkowski",
            # 距离度量类型
            # 可选值："minkowski", "euclidean", "manhattan", "chebyshev", "hamming"等
            # "minkowski"：明可夫斯基距离，当p=2时等价于欧氏距离
            # 其他度量标准适用于特定数据类型

            "n_jobs": -1
            # 并行计算作业数
            # 可选值：-1（使用所有CPU核心），正整数（指定CPU核心数）
            # -1：使用所有可用处理器
            # None：不并行计算
        }

        in_params = params["params"]

        # 参数解析和格式化
        if "n_neighbors" in in_params:
            knn_params["n_neighbors"] = AlgModelParameters.format_to_int(in_params["n_neighbors"])
        if "weights" in in_params:
            knn_params["weights"] = in_params["weights"]
        if "algorithm" in in_params:
            knn_params["algorithm"] = in_params["algorithm"]
        if "leaf_size" in in_params:
            knn_params["leaf_size"] = AlgModelParameters.format_to_int(in_params["leaf_size"])
        if "p" in in_params:
            knn_params["p"] = AlgModelParameters.format_to_int(in_params["p"])
        if "metric" in in_params:
            knn_params["metric"] = in_params["metric"]
        if "n_jobs" in in_params:
            knn_params["n_jobs"] = AlgModelParameters.format_to_int(in_params["n_jobs"])

        print(f"算法参数: {knn_params}")

        # 2. 图片配置参数 - 使用自定义配置
        plot_config = AlgModelParameters.get_image_param("分类模型", "KNN").copy()

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

        # 标准化特征（对KNN非常重要）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        print(f"\n训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")

        # 3. KNN模型训练
        knn = KNeighborsClassifier(
            n_neighbors=knn_params["n_neighbors"],
            weights=knn_params["weights"],
            algorithm=knn_params["algorithm"],
            leaf_size=knn_params["leaf_size"],
            p=knn_params["p"],
            metric=knn_params["metric"],
            n_jobs=knn_params["n_jobs"]
        )
        knn.fit(X_train, y_train)

        # 4. 模型评估
        y_pred = knn.predict(X_test)
        y_train_pred = knn.predict(X_train)

        print("\n" + "=" * 50)
        print("KNN模型训练结果")
        print("=" * 50)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)

        print(f"\n训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        print("\n分类报告:")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report)

        # 5. 生成K值选择曲线图（学习曲线）
        print("\n生成K值选择曲线...")
        k_range = range(1, 31)
        train_scores = []
        test_scores = []

        for k in k_range:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            knn_temp.fit(X_train, y_train)
            train_scores.append(knn_temp.score(X_train, y_train))
            test_scores.append(knn_temp.score(X_test, y_test))

        best_k = k_range[np.argmax(test_scores)]

        plt.figure(figsize=(12, 6))
        plt.plot(k_range, train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=6)
        plt.plot(k_range, test_scores, 's-', label='测试集准确率', linewidth=2, markersize=6)
        plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'最佳K值={best_k}')
        plt.xlabel(plot_config["k_selection_xlabel"], fontsize=12)
        plt.ylabel(plot_config["k_selection_ylabel"], fontsize=12)
        plt.title(plot_config["k_selection_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(k_range[::2])

        k_selection_img_path = os.path.join(output_dir, 'k_selection_curve.png')
        plt.tight_layout()
        plt.savefig(k_selection_img_path, dpi=dpi)
        plt.close()

        # 保存数据
        k_selection_data_path = os.path.join(output_dir, 'k_selection_data.txt')
        k_selection_data = pd.DataFrame({
            'K值': list(k_range),
            '训练集准确率': train_scores,
            '测试集准确率': test_scores
        })
        k_selection_data.to_csv(k_selection_data_path, sep='\t', index=False)

        imgs.append({
            "name": "K值选择曲线",
            "img": k_selection_img_path,
            "data": k_selection_data_path
        })

        # 6. 生成交叉验证K值选择曲线
        print("\n生成交叉验证K值选择曲线...")
        cv_scores = []

        for k in k_range:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_temp, X_scaled, y_encoded, cv=5)
            cv_scores.append(scores.mean())

        cv_best_k = k_range[np.argmax(cv_scores)]

        plt.figure(figsize=(10, 6))
        plt.plot(k_range, cv_scores, 'o-', color='purple', linewidth=2, markersize=6)
        plt.axvline(x=cv_best_k, color='r', linestyle='--', alpha=0.7, label=f'最佳K值={cv_best_k}')
        plt.xlabel(plot_config["cv_k_selection_xlabel"], fontsize=12)
        plt.ylabel(plot_config["cv_k_selection_ylabel"], fontsize=12)
        plt.title(plot_config["cv_k_selection_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(k_range[::2])

        cv_k_selection_img_path = os.path.join(output_dir, 'cv_k_selection_curve.png')
        plt.tight_layout()
        plt.savefig(cv_k_selection_img_path, dpi=dpi)
        plt.close()

        # 保存数据
        cv_k_selection_data_path = os.path.join(output_dir, 'cv_k_selection_data.txt')
        cv_k_selection_data = pd.DataFrame({
            'K值': list(k_range),
            '交叉验证平均准确率': cv_scores
        })
        cv_k_selection_data.to_csv(cv_k_selection_data_path, sep='\t', index=False)

        imgs.append({
            "name": "交叉验证K值选择曲线",
            "img": cv_k_selection_img_path,
            "data": cv_k_selection_data_path
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

        # 8. 生成最近邻距离分布图
        print("\n生成最近邻距离分布...")
        distances, indices = knn.kneighbors(X_test)
        avg_distances = distances.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(avg_distances, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=np.mean(avg_distances), color='red', linestyle='--',
                    linewidth=2, label=f'平均距离: {np.mean(avg_distances):.4f}')
        plt.xlabel(plot_config["distance_dist_xlabel"], fontsize=12)
        plt.ylabel(plot_config["distance_dist_ylabel"], fontsize=12)
        plt.title(plot_config["distance_dist_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        distance_dist_img_path = os.path.join(output_dir, 'distance_distribution.png')
        plt.tight_layout()
        plt.savefig(distance_dist_img_path, dpi=dpi)
        plt.close()

        # 保存数据
        distance_dist_data_path = os.path.join(output_dir, 'distance_distribution_data.txt')
        distance_dist_data = pd.DataFrame({
            '样本索引': range(len(avg_distances)),
            '平均最近邻距离': avg_distances
        })
        distance_dist_data.to_csv(distance_dist_data_path, sep='\t', index=False)

        imgs.append({
            "name": "最近邻距离分布",
            "img": distance_dist_img_path,
            "data": distance_dist_data_path
        })

        # 9. 生成不同权重方法对比图
        print("\n生成权重方法对比...")
        weights_methods = ['uniform', 'distance']
        weight_scores = []

        for weight in weights_methods:
            knn_temp = KNeighborsClassifier(n_neighbors=knn_params["n_neighbors"], weights=weight)
            knn_temp.fit(X_train, y_train)
            weight_scores.append(knn_temp.score(X_test, y_test))

        plt.figure(figsize=(8, 6))
        bars = plt.bar(weights_methods, weight_scores, color=['steelblue', 'lightcoral'], alpha=0.8)
        plt.xlabel(plot_config["weights_comparison_xlabel"], fontsize=12)
        plt.ylabel(plot_config["weights_comparison_ylabel"], fontsize=12)
        plt.title(plot_config["weights_comparison_title"], fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1)

        # 添加数值标签
        for bar, score in zip(bars, weight_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=11)

        weights_comparison_img_path = os.path.join(output_dir, 'weights_comparison.png')
        plt.tight_layout()
        plt.savefig(weights_comparison_img_path, dpi=dpi)
        plt.close()

        # 保存数据
        weights_comparison_data_path = os.path.join(output_dir, 'weights_comparison_data.txt')
        weights_comparison_data = pd.DataFrame({
            '权重方法': weights_methods,
            '准确率': weight_scores
        })
        weights_comparison_data.to_csv(weights_comparison_data_path, sep='\t', index=False)

        imgs.append({
            "name": "不同权重方法对比",
            "img": weights_comparison_img_path,
            "data": weights_comparison_data_path
        })

        # 10. 生成决策边界图（仅适用于2D特征）
        if X.shape[1] == 2:
            print("\n生成决策边界图...")
            fig, ax = plt.subplots(figsize=(10, 8))

            try:
                disp = DecisionBoundaryDisplay.from_estimator(
                    knn,
                    X_train,
                    response_method="predict",
                    alpha=0.5,
                    ax=ax,
                    xlabel=plot_config["decision_boundary_xlabel"],
                    ylabel=plot_config["decision_boundary_ylabel"],
                )

                # 绘制训练数据点
                scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                                     edgecolors='k', s=50, alpha=0.8)

                plt.title(plot_config["decision_boundary_title"], fontsize=16, fontweight='bold')

                decision_boundary_img_path = os.path.join(output_dir, 'decision_boundary.png')
                plt.tight_layout()
                plt.savefig(decision_boundary_img_path, dpi=dpi)
                plt.close()

                imgs.append({
                    "name": "KNN决策边界图",
                    "img": decision_boundary_img_path,
                    "data": ""
                })
            except Exception as e:
                print(f"决策边界图生成失败: {e}")

        # 11. 生成网格搜索热图（如果数据集不是特别大）
        if X.shape[0] < 1000 and len(np.unique(y_encoded)) <= 5:
            print("\n生成网格搜索热图...")
            try:
                param_grid = {
                    'n_neighbors': list(range(3, 16, 2)),
                    'weights': ['uniform', 'distance']
                }

                grid_search = GridSearchCV(
                    KNeighborsClassifier(),
                    param_grid,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)

                # 提取结果
                results = grid_search.cv_results_
                scores = results['mean_test_score'].reshape(
                    len(param_grid['n_neighbors']),
                    len(param_grid['weights'])
                )

                plt.figure(figsize=(10, 8))
                sns.heatmap(scores, annot=True, fmt='.4f', cmap='viridis',
                            xticklabels=param_grid['weights'],
                            yticklabels=param_grid['n_neighbors'])
                plt.xlabel(plot_config["grid_search_xlabel"], fontsize=12)
                plt.ylabel(plot_config["grid_search_ylabel"], fontsize=12)
                plt.title(plot_config["grid_search_title"], fontsize=16, fontweight='bold')

                grid_search_img_path = os.path.join(output_dir, 'grid_search_heatmap.png')
                plt.tight_layout()
                plt.savefig(grid_search_img_path, dpi=dpi)
                plt.close()

                # 保存数据
                grid_search_data_path = os.path.join(output_dir, 'grid_search_data.txt')
                grid_search_data = pd.DataFrame({
                    'K值': np.repeat(param_grid['n_neighbors'], len(param_grid['weights'])),
                    '权重方法': np.tile(param_grid['weights'], len(param_grid['n_neighbors'])),
                    '交叉验证准确率': results['mean_test_score'],
                    '标准差': results['std_test_score']
                })
                grid_search_data.to_csv(grid_search_data_path, sep='\t', index=False)

                imgs.append({
                    "name": "网格搜索热图",
                    "img": grid_search_img_path,
                    "data": grid_search_data_path
                })
            except Exception as e:
                print(f"网格搜索热图生成失败: {e}")

        # 12. 保存模型信息
        model_info = {
            "dataset_info": {
                "samples": df.shape[0],
                "features": df.shape[1] - 1,
                "classes": len(label_encoder.classes_),
                "class_names": label_encoder.classes_.tolist()
            },
            "model_params": knn_params,
            "performance": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "best_k_from_curve": int(best_k),
                "best_k_from_cv": int(cv_best_k),
                "avg_neighbor_distance": float(np.mean(avg_distances) if 'avg_distances' in locals() else 0)
            },
            "classification_report": classification_report(y_test, y_pred,
                                                           target_names=label_encoder.classes_,
                                                           output_dict=True)
        }

        # 保存模型信息到文件
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 13. 生成结果字符串
        result_str += "=======================\n"
        result_str += "KNN模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 成功\n"
        result_str += f"输出目录: {output_dir}\n\n"

        result_str += "=======================\n"
        result_str += "模型总结\n"
        result_str += "=======================\n"
        result_str += f"1. 数据集: {df.shape[0]} 个样本, {df.shape[1] - 1} 个特征\n"
        result_str += f"2. 类别数量: {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})\n"
        result_str += f"3. 使用K值: {knn_params['n_neighbors']}\n"
        result_str += f"4. 权重方法: {knn_params['weights']}\n"
        result_str += f"5. 距离度量: {knn_params['metric']} (p={knn_params['p']})\n"
        result_str += f"6. 训练集准确率: {train_accuracy:.4f}\n"
        result_str += f"7. 测试集准确率: {test_accuracy:.4f}\n"
        result_str += f"8. K值选择最佳值: {best_k}\n"
        result_str += f"9. 交叉验证最佳K值: {cv_best_k}\n"

        if 'avg_distances' in locals():
            result_str += f"10. 平均最近邻距离: {np.mean(avg_distances):.4f}\n"

        # 判断是否过拟合
        if train_accuracy - test_accuracy > 0.1:
            result_str += "11. 模型可能存在过拟合\n"
        else:
            result_str += "11. 模型泛化能力良好\n"

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
        result_str += f"KNN算法参数:\n"
        for key, value in knn_params.items():
            result_str += f"  {key}: {value}\n"
            # 添加参数说明
            if key == "n_neighbors":
                result_str += f"    含义: 最近邻数量，影响模型的复杂度和平滑度\n"
                result_str += f"    范围: 正整数，通常1-20\n"
            elif key == "weights":
                result_str += f"    含义: 权重类型，uniform为等权重，distance为距离反比权重\n"
                result_str += f"    可选值: 'uniform', 'distance'\n"
            elif key == "algorithm":
                result_str += f"    含义: 最近邻搜索算法\n"
                result_str += f"    可选值: 'auto', 'ball_tree', 'kd_tree', 'brute'\n"
            elif key == "leaf_size":
                result_str += f"    含义: 叶节点大小，影响树算法的效率\n"
                result_str += f"    范围: 正整数，默认30\n"
            elif key == "p":
                result_str += f"    含义: 距离度量参数，1为曼哈顿距离，2为欧氏距离\n"
                result_str += f"    范围: 正整数\n"
            elif key == "metric":
                result_str += f"    含义: 距离度量类型\n"
                result_str += f"    可选值: 'minkowski', 'euclidean', 'manhattan'等\n"
            elif key == "n_jobs":
                result_str += f"    含义: 并行计算作业数\n"
                result_str += f"    可选值: -1(使用所有核心), 正整数(指定核心数)\n"

        # 添加图片使用的参数信息
        result_str += f"\n=== 图片使用的参数信息 ===\n"
        for key, value in plot_config.items():
            result_str += f"  {key}: {value}\n"

        print("\n" + result_str)

    except Exception as e:
        error_msg = f"KNN模型运行失败: {e}"
        print(error_msg)
        result_str = "=======================\n"
        result_str += "KNN模型运行结果\n"
        result_str += "=======================\n"
        result_str += f"运行状态: 失败\n"
        result_str += f"错误信息: {str(e)}\n"

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    end_millis = int(time.time() * 1000)
    expend_time = end_millis - start_millis

    print("<<<<<<<<<<<<<<<<<<<< 分类模型 KNN 运行结束 <<<<<<<<<<<<<<<<<<<<")
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
            }
        },
        "start": start_timestamp,
        "end": end_timestamp,
        "expend": expend_time
    }

    # 将结果转换为JSON字符串返回
    return json.dumps(result_json, ensure_ascii=False, indent=2)