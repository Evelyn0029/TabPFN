#  Copyright (c) Prior Labs GmbH 2026.
"""Example of using TabPFN for binary classification.

This example demonstrates how to use TabPFNClassifier on a binary classification task
using the breast cancer dataset from scikit-learn.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))


# 2026.06.05
# 创建调参配置 默认ClassifierTuningConfig=None
tuning_config = ClassifierTuningConfig(
    calibrate_temperature=False,  # 不校准温度
    tune_decision_thresholds=True,   # 启用阈值调优  默认false
    tuning_holdout_frac=0.2,
    tuning_n_folds=5,
)

# 创建模型
model = TabPFNClassifier(
    n_estimators=8,
    tuning_config=tuning_config,
    eval_metric="f1",  # 优化 F1 score
)

# 拟合
model.fit(X_train, y_train)

# 查看调优后的阈值
print(f"Tuned thresholds: {model.tuned_classification_thresholds_}")
# 形状：[n_classes]

# 预测（会自动使用调优后的阈值）
y_pred = model.predict(X_test)
