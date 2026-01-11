
def get_image_param(model_type, model_name):
    if model_type == "分类模型":
        if model_name == "LDA":
            return {
                    "scatter_title": "LDA判别轴投影图",
                    "scatter_xlabel": "判别轴1",
                    "scatter_ylabel": "判别轴2",
                    "decision_title": "LDA决策边界图",
                    "decision_xlabel": "特征1",
                    "decision_ylabel": "特征2",
                    "importance_title": "特征重要性排名",
                    "importance_xlabel": "特征",
                    "importance_ylabel": "重要性得分",
                    "confusion_title": "混淆矩阵热图",
                    "confusion_xlabel": "预测类别",
                    "confusion_ylabel": "真实类别",
                    "scree_title": "LDA判别轴解释方差",
                    "scree_xlabel": "判别轴",
                    "scree_ylabel": "解释方差比例",
                    "cumulative_title": "累计解释方差比例",
                    "cumulative_xlabel": "判别轴数量",
                    "cumulative_ylabel": "累计解释方差比例",
                    "class_means_title": "类别均值图",
                    "class_means_xlabel": "特征",
                    "class_means_ylabel": "均值",
                    "coef_heatmap_title": "判别系数热图",
                    "coef_heatmap_xlabel": "特征",
                    "coef_heatmap_ylabel": "判别函数"
                }
        elif model_name == "SVM":
            return {
                    "scatter_title": "SVM数据分布图（PCA降维）",
                    "scatter_xlabel": "主成分1",
                    "scatter_ylabel": "主成分2",
                    "decision_title": "SVM决策边界图",
                    "decision_xlabel": "特征1",
                    "decision_ylabel": "特征2",
                    "importance_title": "特征重要性排名（线性核）",
                    "importance_xlabel": "特征",
                    "importance_ylabel": "重要性得分",
                    "confusion_title": "SVM混淆矩阵热图",
                    "confusion_xlabel": "预测类别",
                    "confusion_ylabel": "真实类别",
                    "scree_title": "主成分解释方差",
                    "scree_xlabel": "主成分",
                    "scree_ylabel": "解释方差比例",
                    "cumulative_title": "累计解释方差比例",
                    "cumulative_xlabel": "主成分数量",
                    "cumulative_ylabel": "累计解释方差比例",
                    "support_vectors_title": "支持向量分布",
                    "support_vectors_xlabel": "主成分1",
                    "support_vectors_ylabel": "主成分2",
                    "margin_title": "SVM分类边界与间隔",
                    "margin_xlabel": "特征1",
                    "margin_ylabel": "特征2"
                }
        elif model_name == "RF":
            return {
                    "importance_title": "特征重要性排名",
                    "importance_xlabel": "特征",
                    "importance_ylabel": "重要性得分",
                    "confusion_title": "混淆矩阵热图",
                    "confusion_xlabel": "预测类别",
                    "confusion_ylabel": "真实类别",
                    "trees_error_title": "每棵树错误率分布",
                    "trees_error_xlabel": "错误率",
                    "trees_error_ylabel": "频率",
                    "depth_dist_title": "决策树深度分布",
                    "depth_dist_xlabel": "树深度",
                    "depth_dist_ylabel": "树数量",
                    "oob_error_title": "袋外误差随树数量变化",
                    "oob_error_xlabel": "树的数量",
                    "oob_error_ylabel": "袋外误差",
                    "learning_curve_title": "学习曲线",
                    "learning_curve_xlabel": "训练样本比例",
                    "learning_curve_ylabel": "准确率",
                    "perm_importance_title": "排列特征重要性",
                    "perm_importance_xlabel": "特征",
                    "perm_importance_ylabel": "重要性得分",
                    "correlation_title": "特征间相关性热图",
                    "correlation_xlabel": "特征",
                    "correlation_ylabel": "特征",
                    "partial_dependence_title": "部分依赖图",
                    "partial_dependence_xlabel": "特征值",
                    "partial_dependence_ylabel": "预测值",
                    "decision_path_title": "决策路径长度分布",
                    "decision_path_xlabel": "路径长度",
                    "decision_path_ylabel": "样本数量"
                }
    elif model_type == "预测模型":
        if model_name == "KNN":
            return {
                    "k_selection_title": "K值选择曲线",
                    "k_selection_xlabel": "K值",
                    "k_selection_ylabel": "准确率",
                    "cv_k_selection_title": "交叉验证K值选择曲线",
                    "cv_k_selection_xlabel": "K值",
                    "cv_k_selection_ylabel": "平均准确率",
                    "decision_boundary_title": "KNN决策边界图",
                    "decision_boundary_xlabel": "特征1",
                    "decision_boundary_ylabel": "特征2",
                    "confusion_title": "混淆矩阵热图",
                    "confusion_xlabel": "预测类别",
                    "confusion_ylabel": "真实类别",
                    "distance_dist_title": "最近邻距离分布",
                    "distance_dist_xlabel": "距离",
                    "distance_dist_ylabel": "频率",
                    "weights_comparison_title": "不同权重方法对比",
                    "weights_comparison_xlabel": "权重方法",
                    "weights_comparison_ylabel": "准确率",
                    "grid_search_title": "网格搜索热图",
                    "grid_search_xlabel": "K值",
                    "grid_search_ylabel": "权重方法",
                    "precision_recall_title": "精确率-召回率曲线",
                    "precision_recall_xlabel": "召回率",
                    "precision_recall_ylabel": "精确率",
                    "neighbor_distance_title": "每个样本的最近邻距离",
                    "neighbor_distance_xlabel": "样本索引",
                    "neighbor_distance_ylabel": "最近邻距离",
                    "algorithm_comparison_title": "不同算法性能对比",
                    "algorithm_comparison_xlabel": "算法",
                    "algorithm_comparison_ylabel": "准确率",
                    "distance_metric_title": "不同距离度量性能对比",
                    "distance_metric_xlabel": "距离度量",
                    "distance_metric_ylabel": "准确率"
                }
        elif model_name == "BP-ANN":
            return {
                "loss_curve_title": "神经网络训练损失曲线",
                "loss_curve_xlabel": "训练轮次",
                "loss_curve_ylabel": "损失值",
                "accuracy_curve_title": "神经网络训练准确率曲线",
                "accuracy_curve_xlabel": "训练轮次",
                "accuracy_curve_ylabel": "准确率",
                "confusion_title": "混淆矩阵热图",
                "confusion_xlabel": "预测类别",
                "confusion_ylabel": "真实类别",
                "feature_importance_title": "基于权重的特征重要性",
                "feature_importance_xlabel": "特征",
                "feature_importance_ylabel": "重要性得分",
                "layer_weights_title": "神经网络层间权重分布",
                "layer_weights_xlabel": "权重值",
                "layer_weights_ylabel": "频率",
                "activation_dist_title": "隐藏层激活值分布",
                "activation_dist_xlabel": "激活值",
                "activation_dist_ylabel": "频率",
                "gradient_flow_title": "梯度流动可视化",
                "gradient_flow_xlabel": "网络层",
                "gradient_flow_ylabel": "梯度范数",
                "learning_rate_title": "学习率变化曲线",
                "learning_rate_xlabel": "训练轮次",
                "learning_rate_ylabel": "学习率",
                "prediction_error_title": "预测误差分布",
                "prediction_error_xlabel": "预测误差",
                "prediction_error_ylabel": "频率",
                "class_probability_title": "类别概率分布",
                "class_probability_xlabel": "类别",
                "class_probability_ylabel": "平均预测概率"
            }
        elif model_name == "PLSR":
            return {
                    "scree_plot_title": "PLS成分数选择 - 碎石图",
                    "scree_plot_xlabel": "PLS成分数",
                    "scree_plot_ylabel": "交叉验证得分",
                    "scatter_train_title": "训练集: 预测值 vs 实际值",
                    "scatter_train_xlabel": "实际值",
                    "scatter_train_ylabel": "预测值",
                    "scatter_test_title": "测试集: 预测值 vs 实际值",
                    "scatter_test_xlabel": "实际值",
                    "scatter_test_ylabel": "预测值",
                    "vip_scores_title": "特征重要性 - VIP分数",
                    "vip_scores_xlabel": "特征",
                    "vip_scores_ylabel": "VIP分数",
                    "coefficients_title": "PLS模型系数",
                    "coefficients_xlabel": "特征",
                    "coefficients_ylabel": "系数值",
                    "residuals_title": "残差分布图",
                    "residuals_xlabel": "残差",
                    "residuals_ylabel": "频率",
                    "qq_plot_title": "Q-Q图（残差正态性检验）",
                    "qq_plot_xlabel": "理论分位数",
                    "qq_plot_ylabel": "样本分位数",
                    "actual_vs_pred_title": "实际值 vs 预测值时序图",
                    "actual_vs_pred_xlabel": "样本索引",
                    "actual_vs_pred_ylabel": "数值"
                }
        elif model_name == "MLR":
            return {
                    "scatter_train_title": "训练集: 预测值 vs 实际值",
                    "scatter_train_xlabel": "实际值",
                    "scatter_train_ylabel": "预测值",
                    "scatter_test_title": "测试集: 预测值 vs 实际值",
                    "scatter_test_xlabel": "实际值",
                    "scatter_test_ylabel": "预测值",
                    "coefficients_title": "多元线性回归模型系数",
                    "coefficients_xlabel": "特征",
                    "coefficients_ylabel": "系数值",
                    "residuals_title": "残差分布图",
                    "residuals_xlabel": "残差",
                    "residuals_ylabel": "频率",
                    "qq_plot_title": "Q-Q图（残差正态性检验）",
                    "qq_plot_xlabel": "理论分位数",
                    "qq_plot_ylabel": "样本分位数",
                    "actual_vs_pred_title": "实际值 vs 预测值时序图",
                    "actual_vs_pred_xlabel": "样本索引",
                    "actual_vs_pred_ylabel": "数值",
                    "scree_plot_title": "主成分分析 - 碎石图",
                    "scree_plot_xlabel": "主成分",
                    "scree_plot_ylabel": "方差解释率",
                    "cumulative_variance_title": "累计方差解释率",
                    "cumulative_variance_xlabel": "主成分数量",
                    "cumulative_variance_ylabel": "累计方差解释率",
                    "residuals_vs_fitted_title": "残差 vs 拟合值图",
                    "residuals_vs_fitted_xlabel": "拟合值",
                    "residuals_vs_fitted_ylabel": "残差",
                    "feature_correlation_title": "特征相关性热图",
                    "feature_correlation_xlabel": "特征",
                    "feature_correlation_ylabel": "特征",
                    "prediction_error_title": "预测误差图",
                    "prediction_error_xlabel": "实际值",
                    "prediction_error_ylabel": "预测误差"
                }
        elif model_name == "SVR":
            return {
                    "scatter_train_title": "SVR训练集: 预测值 vs 实际值",
                    "scatter_train_xlabel": "实际值",
                    "scatter_train_ylabel": "预测值",
                    "scatter_test_title": "SVR测试集: 预测值 vs 实际值",
                    "scatter_test_xlabel": "实际值",
                    "scatter_test_ylabel": "预测值",
                    "residuals_title": "SVR残差分布图",
                    "residuals_xlabel": "残差",
                    "residuals_ylabel": "频率",
                    "qq_plot_title": "SVR Q-Q图（残差正态性检验）",
                    "qq_plot_xlabel": "理论分位数",
                    "qq_plot_ylabel": "样本分位数",
                    "actual_vs_pred_title": "SVR实际值 vs 预测值时序图",
                    "actual_vs_pred_xlabel": "样本索引",
                    "actual_vs_pred_ylabel": "数值",
                    "scree_plot_title": "SVR主成分分析 - 碎石图",
                    "scree_plot_xlabel": "主成分",
                    "scree_plot_ylabel": "方差解释率",
                    "cumulative_variance_title": "SVR累计方差解释率",
                    "cumulative_variance_xlabel": "主成分数量",
                    "cumulative_variance_ylabel": "累计方差解释率",
                    "residuals_vs_fitted_title": "SVR残差 vs 拟合值图",
                    "residuals_vs_fitted_xlabel": "拟合值",
                    "residuals_vs_fitted_ylabel": "残差",
                    "feature_correlation_title": "SVR特征相关性热图",
                    "feature_correlation_xlabel": "特征",
                    "feature_correlation_ylabel": "特征",
                    "prediction_error_title": "SVR预测误差图",
                    "prediction_error_xlabel": "实际值",
                    "prediction_error_ylabel": "预测误差",
                    "support_vectors_title": "SVR支持向量分布",
                    "support_vectors_xlabel": "特征1（主成分1）",
                    "support_vectors_ylabel": "特征2（主成分2）",
                    "epsilon_tube_title": "SVR ε-不敏感带示意图",
                    "epsilon_tube_xlabel": "特征",
                    "epsilon_tube_ylabel": "目标值",
                    "learning_curve_title": "SVR学习曲线",
                    "learning_curve_xlabel": "训练样本数量",
                    "learning_curve_ylabel": "得分",
                    "hyperparameter_heatmap_title": "SVR超参数热力图",
                    "hyperparameter_heatmap_xlabel": "C参数",
                    "hyperparameter_heatmap_ylabel": "γ参数"
                }
    elif model_type == "降维模型":
        if model_name == "PCA":
            return {
                    "scree_plot_title": "PCA碎石图 - 方差解释比例",
                    "scree_plot_xlabel": "主成分",
                    "scree_plot_ylabel": "方差解释比例",
                    "cumulative_variance_title": "累计方差解释比例",
                    "cumulative_variance_xlabel": "主成分数量",
                    "cumulative_variance_ylabel": "累计方差解释比例",
                    "pca_2d_scatter_title": "PCA二维散点图",
                    "pca_2d_scatter_xlabel": "主成分1",
                    "pca_2d_scatter_ylabel": "主成分2",
                    "pca_3d_scatter_title": "PCA三维散点图",
                    "pca_3d_scatter_xlabel": "主成分1",
                    "pca_3d_scatter_ylabel": "主成分2",
                    "pca_3d_scatter_zlabel": "主成分3",
                    "biplot_title": "PCA双标图",
                    "biplot_xlabel": "主成分1",
                    "biplot_ylabel": "主成分2",
                    "loading_plot_title": "主成分载荷图",
                    "loading_plot_xlabel": "特征",
                    "loading_plot_ylabel": "载荷值",
                    "correlation_circle_title": "相关性圆圈图",
                    "correlation_circle_xlabel": "主成分1",
                    "correlation_circle_ylabel": "主成分2",
                    "heatmap_title": "PCA主成分热图",
                    "heatmap_xlabel": "特征",
                    "heatmap_ylabel": "主成分",
                    "variance_ratio_table_title": "方差解释比例表",
                    "pairplot_title": "主成分对图",
                    "pairplot_diag": "直方图",
                    "pairplot_offdiag": "散点图"
                }
        elif model_name == "LDA":
            return {
                    "lda_2d_scatter_title": "LDA二维投影结果",
                    "lda_2d_scatter_xlabel": "线性判别式 1",
                    "lda_2d_scatter_ylabel": "线性判别式 2",
                    "lda_3d_scatter_title": "LDA三维投影结果",
                    "lda_3d_scatter_xlabel": "线性判别式 1",
                    "lda_3d_scatter_ylabel": "线性判别式 2",
                    "lda_3d_scatter_zlabel": "线性判别式 3",
                    "explained_variance_title": "LDA判别式解释方差比例",
                    "explained_variance_xlabel": "线性判别式",
                    "explained_variance_ylabel": "解释方差比例",
                    "separation_metrics_title": "LDA类别分离度评估",
                    "separation_metrics_xlabel": "类别",
                    "separation_metrics_ylabel": "分离度指标",
                    "discriminant_vectors_title": "LDA判别向量（前10个特征）",
                    "discriminant_vectors_xlabel": "特征",
                    "discriminant_vectors_ylabel": "判别向量值",
                    "fisher_score_title": "特征Fisher判别得分",
                    "fisher_score_xlabel": "特征",
                    "fisher_score_ylabel": "Fisher得分",
                    "class_centroids_title": "LDA类别中心投影",
                    "class_centroids_xlabel": "线性判别式 1",
                    "class_centroids_ylabel": "线性判别式 2",
                    "mahalanobis_distance_title": "类别间马氏距离热图",
                    "mahalanobis_distance_xlabel": "类别",
                    "mahalanobis_distance_ylabel": "类别",
                    "before_after_comparison_title": "LDA降维前后对比",
                    "before_after_comparison_xlabel": "原始特征1",
                    "before_after_comparison_ylabel": "原始特征2"
                }
        elif model_name == "LLE":
            return {
                    "lle_2d_scatter_title": "LLE二维嵌入结果",
                    "lle_2d_scatter_xlabel": "嵌入维度1",
                    "lle_2d_scatter_ylabel": "嵌入维度2",
                    "lle_3d_scatter_title": "LLE三维嵌入结果",
                    "lle_3d_scatter_xlabel": "嵌入维度1",
                    "lle_3d_scatter_ylabel": "嵌入维度2",
                    "lle_3d_scatter_zlabel": "嵌入维度3",
                    "neighbors_analysis_title": "邻居数对LLE结果的影响",
                    "neighbors_analysis_xlabel": "邻居数",
                    "neighbors_analysis_ylabel": "重构误差",
                    "reconstruction_error_title": "LLE重构误差分布",
                    "reconstruction_error_xlabel": "样本索引",
                    "reconstruction_error_ylabel": "重构误差",
                    "eigenvalue_spectrum_title": "LLE特征值谱",
                    "eigenvalue_spectrum_xlabel": "特征值索引",
                    "eigenvalue_spectrum_ylabel": "特征值",
                    "neighborhood_graph_title": "最近邻图可视化",
                    "neighborhood_graph_xlabel": "特征1",
                    "neighborhood_graph_ylabel": "特征2",
                    "parameter_sensitivity_title": "LLE参数敏感性分析",
                    "parameter_sensitivity_xlabel": "参数值",
                    "parameter_sensitivity_ylabel": "重构误差",
                    "manifold_unfolding_title": "流形展开过程",
                    "manifold_unfolding_xlabel": "迭代步骤",
                    "manifold_unfolding_ylabel": "嵌入坐标"
                }
    return {}

def get_param(model_type, model_name):
    """根据模型类型和名称获取参数定义"""
    if model_type == "分类模型":
        if model_name == "LDA":  # 分类模型 LDA
            return [
                {
                    "name": "solver", "label": "求解算法", "type": "choice",
                    "options": ["svd", "lsqr", "eigen"], "default": "svd",
                    "tip": "solver\n数据维度 < 样本数: 使用 'svd'\n需要正则化: 使用 'lsqr' 或 'eigen'\n大数据集: 使用 'lsqr'"
                },
                {
                    "name": "shrinkage", "label": "收缩参数", "type": "choice",
                    "options": ["None", "auto", "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                    "default": "None",
                    "tip":"shrinkage\nNone: 不使用收缩\n'auto': 自动使用 Ledoit-Wolf 方法计算最优收缩系数\n0.0: 无收缩\n1.0: 最大收缩\n仅在 solver 为 'lsqr' 或 'eigen' 时有效"
                },
                {
                    "name":"n_components", "label": "降维维度", "type": "choice",
                    "options": ["None", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "default": "None",
                    "tip":"n_components\nNone: 自动设置为 min(n_classes-1, n_features)\n整数: 指定要保留的判别成分数量"
                },
                {
                    "name":"store_covariance", "label": "存储协方差矩阵", "type": "bool",
                    "options": [],
                    "default": False,
                    "tip":"store_covariance\nFalse: 不存储协方差矩阵，节省内存\nTrue: 存储协方差矩阵，可用于后续分析"
                }
            ]
        elif model_name == "SVM":  # 分类模型 SVM
            return [
                {
                    "name": "kernel", "label": "核函数", "type": "choice",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                    "default": "rbf",
                    "tip":"kernel\nlinear（线性核）：适用于线性可分的数据\n公式：K(x, y) = x · y\n特点：计算速度快，参数少，不易过拟合\n适用场景：数据量较大，特征数量较多的情况\n"
                          "rbf（径向基核，默认）：最常用的非线性核函数\n公式：K(x, y) = exp(-γ * ||x - y||²)\n特点：能够处理非线性关系，通用性好\n适用场景：大多数情况下首选，尤其是类别边界复杂时\n"
                          "poly（多项式核）：处理多项式关系\n公式：K(x, y) = (γ * x · y + r)^d\n特点：可以控制多项式的阶数\n适用场景：特征之间存在多项式关系的情况\n"
                          "sigmoid（S形核）：类似神经网络的激活函数\n公式：K(x, y) = tanh(γ * x · y + r)\n特点：在某些情况下效果较好\n适用场景：特定类型的数据分类"
                },
                {
                    "name": "C", "label": "正则化参数", "type": "choice",
                    "options": ["0.001", "0.01", "0.1", "1.0", "10", "100", "1000"],
                    "default": "1.0",
                    "tip":"C\n控制模型对分类错误的惩罚程度\n通常为 0.001 到 1000 之间的浮点数\n先尝试默认值,如果过拟合，减小C值,如果欠拟合，增大C值"
                },
                {
                    "name": "gamma", "label": "核系数", "type": "choice",
                    "options":["scale", "auto", "0.001", "0.01", "0.1", "1.0", "10"],
                    "default": "scale",
                    "tip":"gamma\n影响RBF核函数中单个训练样本的影响范围\n"
                          "scale（默认）：自动计算\n公式：γ = 1 / (n_features * X.var())\n根据特征数量和方差自动调整\n"
                          "auto：简化的自动计算\n公式：γ = 1 / n_features\n只考虑特征数量"
                }
            ]
        elif model_name == "RF":  # 分类模型 RF
            return [
                {
                    "name": "n_estimators", "label": "决策树数量", "type": "int", "min": 1, "max": 500,
                    "options": [],
                    "default": 100,
                    "tip": "n_estimators\n决定森林中决策树的数量\n值越大：模型越稳定，但计算时间越长\n值越小：计算速度快，但可能欠拟合\n建议范围：50-500"
                },
                {
                    "name": "max_depth", "label": "最大深度", "type": "int", "min":1, "max" : 30,
                    "options": [],
                    "default": 3,
                    "tip": "max_depth\n每棵树的最大深度\nNone：不限制深度，直到所有叶子节点纯净\n值较小：防止过拟合，模型更简单\n值较大：模型更复杂，可能过拟合\n建议范围：3-30"
                },
                {
                    "name": "min_samples_split", "label": "最小分裂样本数", "type": "int", "min" : 1, "max" : 20,
                    "options": [],
                    "default": 2,
                    "tip": "min_samples_split\n内部节点再划分所需最小样本数\n值较小：树更复杂，可能过拟合\n值较大：防止过拟合，模型更简单\n建议范围：2-20"
                },
                {
                    "name": "min_samples_leaf", "label": "最小叶子样本数", "type": "int", "min": 1, "max": 10,
                    "options": [],
                    "default": 1,
                    "tip": "min_samples_leaf\n叶节点最少样本数\n值较小：对噪声敏感，可能过拟合\n值较大：模型更稳定，防止过拟合\n建议范围：1-10"
                },
                {
                    "name": "max_features", "label": "最大特征数", "type": "choice",
                    "options": ["sqrt", "log2", "auto", "None"],
                    "default": "sqrt",
                    "tip": "max_features\n寻找最佳分割时考虑的特征数量\n'sqrt'：sqrt(n_features)\n'log2'：log2(n_features)\n'auto'：sqrt(n_features)\n'None'：使用所有特征"
                },
                {
                    "name": "bootstrap",
                    "label": "Bootstrap采样",
                    "type": "bool",
                    "options": [],
                    "default": True,
                    "tip": "bootstrap\n是否使用bootstrap采样\nTrue：使用bootstrap采样，增加模型多样性\nFalse：使用整个数据集，减少偏差"
                },
                {
                    "name": "random_state", "label": "随机种子", "type": "int",
                    "min":1, "max": 200,
                    "options": [],
                    "default": 42,
                    "tip": "random_state\n控制随机性的参数\n固定值：确保结果可重复\nNone：每次运行结果不同\n用于调试和结果复现"
                }
            ]
    elif model_type == "预测模型":
        if model_name == "KNN":  # 预测模型 KNN
            return [
                {
                    "name": "n_neighbors", "label": "K值，最近邻的数量", "type": "int",
                    "min":1, "max": 20,
                    "options": [],
                    "default": 5,
                    "tip":"n_neighbors\n可选值范围：正整数，通常1-20之间，默认5\n太小可能导致过拟合，太大可能导致欠拟合"
                },
                {
                    "name":"weights", "label": "权重类型", "type": "choice",
                    "options": ["uniform", "distance"],
                    "default": "uniform",
                    "tip": "weights\nuniform(均值权重)：所有近邻的权重相等\ndistance(距离反比权重)：权重与距离成反比，距离越近权重越大"
                },
                {
                    "name": "algorithm", "label": "计算最近邻的算法", "type": "choice",
                    "options": ["auto", "ball_tree", "kd_tree", "brute"],
                    "default": "auto",
                    "tip": "algorithm\nauto:根据数据自动选择最合适的算法\nball_tree:适用于高维数据\nkd_tree:适用于低维数据（维度<20）\nbrute:暴力搜索，适用于小数据集"
                },
                {
                    "name": "leaf_size", "label": "叶节点大小", "type": "int",
                    "min":1, "max": 2000,
                    "default": 30,
                    "tip": "leaf_size\n叶节点大小（仅对ball_tree和kd_tree有效）\n可选值范围：正整数，默认30\n影响树构建的速度和内存使用"
                },
                {
                    "name": "p", "label": "距离度量参数", "type": "int",
                    "min": 1, "max": 2000,
                    "default": 2,
                    "tip": "p\n可选值：1（曼哈顿距离），2（欧氏距离），或其他正整数\np=1：曼哈顿距离（L1距离）\np=2：欧氏距离（L2距离）\np>2：明可夫斯基距离"
                },
                {
                    "name": "metric", "label": "距离度量类型", "type": "choice",
                    "options": ["minkowski", "euclidean", "manhattan", "chebyshev", "hamming"],
                    "default": "minkowski",
                    "tip": "metric\nminkowski：明可夫斯基距离，当p=2时等价于欧氏距离\n其他度量标准适用于特定数据类型"
                },
                {
                    "name": "n_jobs", "label": "并行计算作业数", "type": "choice",
                    "options": ["-1", "None", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20","21" ,"22", "23", "24" ,"25", "26", "27", "28", "29", "30", "31", "32"],
                    "default": "-1",
                    "tip": "n_jobs\n-1：使用所有可用处理器\nNone：不并行计算"
                }
            ]
        elif model_name == "BP-ANN":  # 预测模型 BP-ANN
            return [
                {
                    "name": "activation", "label": "激活函数", "type": "choice",
                    "options": ["identity", "logistic", "tanh", "relu"],
                    "default": "relu",
                    "tip":"activation\n隐藏层的激活函数类型\n"
                          "identity: 线性激活函数 f(x) = x\n"
                          "logistic: Logistic sigmoid函数，f(x) = 1 / (1 + exp(-x))\n"
                          "tanh: 双曲正切函数，f(x) = tanh(x)\n"
                          "relu: 修正线性单元，f(x) = max(0, x)（默认）"
                },
                {
                    "name": "solver", "label": "优化算法", "type": "choice",
                    "options": ["lbfgs", "sgd", "adam"],
                    "default": "adam",
                    "tip": "solver\n用于权重优化的求解器\n"
                           "lbfgs: 准牛顿方法，适合小数据集\n"
                           "sgd: 随机梯度下降\n"
                           "adam: 基于随机梯度的优化器，适合大数据集（默认）"
                },
                {
                    "name": "learning_rate", "label": "学习率策略", "type": "choice",
                    "options": ["constant", "invscaling", "adaptive"],
                    "default": "constant",
                    "tip": "learning_rate\n学习率的更新策略\nconstant: 恒定学习率（默认）\n"
                           "invscaling: 随时间逐渐减小，learning_rate_init / t^power_t\n"
                           "adaptive: 当训练损失不再下降时减小学习率"
                },
                {
                    "name": "learning_rate_init", "label": "初始学习率", "type": "float",
                    "min": 0.0001, "max": 0.1, "decimals":4,
                    "default": 0.001,
                    "tip": "learning_rate_init\n取值范围：正浮点数，通常0.0001到0.1"

                },
                {
                    "name": "shuffle", "label": "是否打乱数据", "type": "bool",
                    "options": [],
                    "default": True,
                    "tip": "shuffle\n是否打乱数据\n含义：是否在每个epoch打乱训练样本"
                }
            ]
        elif model_name == "PLSR":  # 预测模型 PLSR
            return [
                {
                    "name": "target_sensor", "label": "目标列名", "type": "line",
                    "default": "",
                    "tip": "target_sensor\n要预测的目标变量列名"
                },
                {
                    "name": "max_iter", "label": "最大迭代次数", "type": "int",
                    "min": 1, "max": 1000,
                    "options": [],
                    "default": 500,
                    "tip":"max_iter\nNIPALS算法的最大迭代次数\n取值范围：正整数，通常100-1000"
                },
                {
                    "name":"copy", "label": "是否复制数据", "type": "bool",
                    "options": [],
                    "default": True,
                    "tip":"copy\n是否复制X和Y，如果为False则可能覆盖原始数据\n"
                },
                {
                    "name": "test_size", "label": "测试集比例", "type": "float",
                    "min": 0, "max": 1,
                    "options": [],
                    "default": 0.2,
                    "tip": "test_size\n测试集占总数据的比例\n取值范围：0到1之间的浮点数"
                },
                {
                    "name": "vip_threshold", "label": "VIP分数阈值", "type": "float",
                    "min": 0, "max": 10,
                    "options": [],
                    "default": 1.0,
                    "tip": "vip_threshold\n用于判断特征重要性的VIP分数阈值\n取值范围：正浮点数，通常0.8-1.5"
                }
            ]
        elif model_name == "MLR":  # 预测模型 MLR
            return [
                {
                    "name": "target_column", "label": "目标列名", "type": "line",
                    "default": "",
                    "tip": "target_column\n要预测的目标变量列名"
                },
                {
                    "name": "test_size", "label": "测试集比例", "type": "float",
                    "min": 0, "max": 1,
                    "options": [],
                    "default": 0.2,
                    "tip": "test_size\n测试集占总数据的比例\n取值范围：0到1之间的浮点数"
                },
                {
                    "name": "cv_folds", "label": "交叉验证折数", "type": "int",
                    "min": 1, "max": 10,
                    "options": [],
                    "default": 3,
                    "tip": "cv_folds\n取值范围：正整数，通常3-10"
                },
                {
                    "name": "standardize", "label": "是否标准化特征", "type": "bool",
                    "options": [],
                    "default": True,
                    "tip": "standardize\n取值范围：布尔值True或False"
                },
                {
                    "name": "fit_intercept", "label": "是否计算截距", "type": "bool",
                    "options": [],
                    "default": True,
                    "tip": "fit_intercept\n取值范围：布尔值True或False"
                },
                {
                    "name": "positive", "label": "强制系数为正", "type": "bool",
                    "options": [],
                    "default": False,
                    "tip": "positive\n是否强制系数为正\n取值范围：布尔值True或False"
                }
            ]
        elif model_name == "SVR":  # 预测模型 SVR
            return [
                {
                    "name": "target_column", "label": "目标列名", "type": "line",
                    "default": "",
                    "tip": "target_column\n要预测的目标变量列名"
                },
                {
                    "name": "test_size", "label": "测试集比例", "type": "float",
                    "min": 0, "max": 1,
                    "options": [],
                    "default": 0.2,
                    "tip": "test_size\n测试集占总数据的比例\n取值范围：0到1之间的浮点数"
                },
                {
                    "name": "cv_folds", "label": "交叉验证折数", "type": "int",
                    "min": 1, "max": 10,
                    "options": [],
                    "default": 3,
                    "tip": "cv_folds\n取值范围：正整数，通常3-10"
                },
                {
                    "name": "standardize", "label": "是否标准化特征", "type": "bool",
                    "options": [],
                    "default": True,
                    "tip": "standardize\n取值范围：布尔值True或False"
                },
                {
                    "name": "kernel", "label": "核函数类型", "type": "choice",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                    "default": "rbf",
                    "tip":"kernel\nlinear: 线性核函数，适用于线性可分问题，计算速度快\n"
                          "poly: 多项式核函数，适用于非线性问题，可调参数度(degree)\n"
                          "rbf: 径向基函数核（默认），适用于大多数非线性问题，泛化能力强\n"
                          "sigmoid: sigmoid核函数，类似于神经网络激活函数"
                },
                {
                    "name": "gamma", "label": "核函数系数", "type": "choice",
                    "options": ["scale", "auto", "0.01", "0.1", "1", "10"],
                    "default": "scale",
                    "tip": "gamma\n"
                           "scale: 1 / (n_features * X.var())，推荐用于特征方差不同的情况\n"
                           "auto: 1 / n_features，旧版本默认值\n"
                           "浮点数: 自定义gamma值，例如 0.01, 0.1, 1, 10\n"
                           "- 较小的gamma：决策边界更平滑，模型更简单\n"
                           "- 较大的gamma：决策边界更复杂，模型更复杂"
                }
            ]
    elif model_type == "降维模型":
        if model_name == "PCA":  # 降维模型 PCA
            return [
                {
                    "name": "target_column", "label": "目标列名", "type": "line",
                    "default": "target",
                    "tip": "target_column\n"
                           "用于着色\n"
                           "注意：如果数据没有目标列，可以设置为None，则所有点使用相同颜色"
                },
                {
                    "name": "n_components", "label": "主成分数量", "type": "choice",
                    "options":["None", "mle", "2", "3", "5", "0.80", "0.95"],
                    "default": "None",
                    "tip": "n_components\n主成分数量\n"
                           "None: 保留所有主成分\n"
                           "整数 (如 2, 3, 5): 保留指定数量的主成分\n"
                           "浮点数 (如 0.95, 0.80): 保留累计解释方差达到该比例的主成分\n"
                           "'mle': 使用MLE（最大似然估计）自动选择主成分数量"
                },
                {
                    "name": "whiten", "label": "是否白化", "type": "bool",
                    "default": False,
                    "tip":"whiten\n注意：白化通常用于某些算法（如K-means）的预处理\n"
                          "True: 白化数据，使各主成分具有单位方差\n"
                          "False: 保留原始尺度（默认）"
                },
                {
                    "name":"svd_solver", "label": "SVD求解器", "type": "choice",
                    "options":["auto", "full", "arpack", "randomized"],
                    "default": "auto",
                    "tip":"svd_solver\nSVD求解器\n"
                          "auto: 自动选择（默认），根据数据大小和n_components参数自动选择\n"
                          "full: 完整的SVD（LAPACK），适用于所有情况但计算成本高\n"
                          "arpack: 截断SVD，适用于n_components较小的情况\n"
                          "randomized: 随机SVD，适用于大数据集，牺牲精度换取速度"
                }
            ]
        elif model_name == "LDA":  # 降维模型 LDA
            return [
                {
                    "name": "target_column", "label": "目标列名", "type": "line",
                    "default": "target",
                    "tip": "target_column\n"
                           "目标列名（必须提供，LDA是监督降维方法）\n"
                           "注意：LDA必须有目标变量，且目标变量应该是类别型"
                },
                {
                    "name": "solver", "label": "求解器", "type": "choice",
                    "options": ["svd", "lsqr", "eigen"],
                    "default": "svd",
                    "tip":"solver\n求解器\n"
                          "svd: 奇异值分解（默认），无需计算协方差矩阵\n"
                          "lsqr: 最小二乘解，可用于降维\n"
                          "eigen: 特征值分解，适用于分类任务"
                }
            ]
        elif model_name == "LLE":  # 降维模型 LLE
            return [
                {
                    "name": "target_column", "label": "目标列名", "type": "line",
                    "default": "target",
                    "tip": "target_column\n"
                           "用于着色\n"
                           "注意：如果数据没有目标列，可以设置为None，则所有点使用相同颜色"
                },
                {
                    "name": "n_components", "label": "目标维度", "type": "int",
                    "min": 1, "max": 200,
                    "default": 2,
                    "tip": "n_components\n取值范围\n"
                           "整数 (如 2, 3): 降维到指定维度\n"
                           "通常用于可视化选择2或3维\n"
                           "默认值：2"
                },
                {
                    "name": "n_neighbors", "label": "邻居数", "type": "int",
                    "min": 1, "max": 200,
                    "default": 10,
                    "tip": "n_neighbors\n邻居数（局部邻域大小）\n"
                           "取值范围：正整数，通常5-20\n"
                           "- 过小：可能无法捕捉局部结构，导致不稳定的嵌入\n"
                           "- 过大：可能破坏局部线性假设，失去非线性特性\n"
                           "默认值：10"
                },
                {
                    "name": "eigen_solver", "label": "特征值求解器", "type": "choice",
                    "options":["auto", "arpack", "dense"],
                    "default": "auto",
                    "tip":"eigen_solver\n特征值求解器\n"
                          "auto: 自动选择（默认）\n"
                          "arpack: 使用ARPACK求解器，适用于大规模数据\n"
                          "dense: 使用密集矩阵求解器，适用于小规模数据\n"
                          "默认值：'auto'"
                },
                {
                    "name": "method", "label": "LLE方法变体", "type": "choice",
                    "options": ["standard", "modified", "hessian", "ltsa"],
                    "default": "standard",
                    "tip": "standard\nLLE方法变体\n"
                           "standard: 标准LLE（默认）\n"
                           "modified: 修改的LLE，增加稳定性\n"
                           "hessian: Hessian LLE，关注局部曲率\n"
                           "ltsa: 局部切空间对齐，保持全局几何结构"
                },
                {
                    "name": "neighbors_algorithm", "label": "最近邻算法", "type": "choice",
                    "options":["auto", "brute", "kd_tree", "ball_tree"],
                    "default": "auto",
                    "tip":"neighbors_algorithm\n最近邻算法\n"
                          "auto: 自动选择（默认）\n"
                          "brute: 暴力搜索\n"
                          "kd_tree: KD树算法\n"
                          "ball_tree: 球树算法"
                }
            ]

    return []

def format_to_int (value):
    if isinstance(value, int):
        return value
    if value == "None":
        return None
    elif value == "auto":
        return "auto"
    else:
        return int(value)

def format_to_float(value):
    if isinstance(value, float):
        return value
    if value == "None":
        return None
    elif value == "auto":
        return "auto"
    elif value == "scale":
        return "scale"
    else:
        return float(value)
