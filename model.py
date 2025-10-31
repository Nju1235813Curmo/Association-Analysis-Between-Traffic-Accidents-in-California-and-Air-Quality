import os
import pickle
import pandas as pd
import gc
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

current_dir = os.path.dirname(os.path.abspath(__file__))
middle_data_path = os.path.join(current_dir, "ca_data_full.pkl")

try:
    with open(middle_data_path, 'rb') as f:
        middle_data = pickle.load(f)
    ca_accidents = middle_data['ca_accidents']
    ca_air_quality = middle_data['ca_air_quality']
    raw_acc_path = middle_data['raw_accident_path']
    print(f"读取中间数据：ca_accidents({len(ca_accidents)}行, {len(ca_accidents.columns)}列)")
except Exception as e:
    print(f"读取中间数据失败：{str(e)}，请先运行analysis_part1.py")
    exit()

# 按需补充Dash需要的原始字段
dash_needed_fields = ['Start_Time', 'End_Time']
missing_fields = [f for f in dash_needed_fields if f not in ca_accidents.columns]

if missing_fields:
    print(f"⚠ 缺失Dash所需字段：{missing_fields}，将补充读取原始文件...")
    raw_supplement = pd.read_csv(
        raw_acc_path,
        usecols=['ID'] + missing_fields
    )
    ca_accidents = pd.merge(
        ca_accidents,
        raw_supplement,
        on='ID',
        how='left'
    )
    print(f"已补充字段：{missing_fields}，当前总字段数：{len(ca_accidents.columns)}")
    del raw_supplement
    gc.collect()


# ==================== 10. 模型对比分析====================
print("\n" + "="*80)
print("10. 模型对比分析")
print("="*80)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
import time
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 获取保存路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
def get_save_path(filename):
    return os.path.join(current_dir, filename)

# ---------------------- 1. 数据准备----------------------
try:
    assert 'ca_accidents' in locals(), "未找到事故数据(ca_accidents)"
    assert len(ca_accidents) > 0, "事故数据为空"
    
    # 特征列
    features = []
    if 'Visibility(mi)' in ca_accidents.columns:
        features.append('Visibility(mi)')
    if 'Temperature(F)' in ca_accidents.columns:
        features.append('Temperature(F)')
    if 'Humidity(%)' in ca_accidents.columns:
        features.append('Humidity(%)')
    if 'Wind_Speed(mph)' in ca_accidents.columns:
        features.append('Wind_Speed(mph)')
    features.extend(['Hour', 'Month', 'DayOfWeek'])
    available_features = [f for f in features if f in ca_accidents.columns]
    assert len(available_features) >= 2, "可用特征不足（至少需要2个特征）"
    
    # 目标变量
    ca_accidents['Severe'] = (ca_accidents['Severity'] >= 3).astype(int)
    model_data = ca_accidents[available_features + ['Severe']].dropna()
    assert len(model_data) > 100, "有效样本数不足（至少需要100个样本）"
    assert len(model_data['Severe'].unique()) >= 2, "目标变量类别单一，无法训练分类模型"
    
    # 数据拆分与标准化
    X = model_data[available_features]
    y = model_data['Severe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    del ca_accidents, model_data, X, y
    gc.collect()
    print("   ✔ 已删除原始数据冗余变量，释放内存")

except AssertionError as e:
    print(f"   ✗ {str(e)}，跳过模型对比分析")
    print("\n" + "="*80)
    print("模型对比分析完成！")
    print(f"生成图表路径: {current_dir}")
    print("="*80)
    exit()

except Exception as e:
    print(f"   ✗ 数据准备失败: {str(e)}，跳过模型对比分析")
    print("\n" + "="*80)
    print("模型对比分析完成！")
    print(f"生成图表路径: {current_dir}")
    print("="*80)
    exit()

# ---------------------- 2. 模型训练----------------------
models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),  # n_jobs=1避免内存峰值
    '决策树': DecisionTreeClassifier(random_state=42, max_depth=10),
    '朴素贝叶斯': GaussianNB()
}

# 存储结果
results = []
model_predictions = {}
model_probabilities = {}
model_objects = {}
total_start_time = time.time()

print("\n正在训练和评估各模型...")
for name, model in models.items():
    print(f"\n训练 {name}...")
    start_time = time.time()
    try:
        # 训练模型
        model.fit(X_train_scaled, y_train)
        model_objects[name] = model
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.5
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        del cv_scores
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 保存结果
        results.append({
            '模型': name,
            '准确率': accuracy, '精确率': precision, '召回率': recall,
            'F1分数': f1, 'AUC': auc,
            '交叉验证均值': cv_mean, '交叉验证标准差': cv_std,
            '训练时间(秒)': time.time() - start_time,
            '真阳性': cm[1,1] if len(cm)>1 else 0, '真阴性': cm[0,0],
            '假阳性': cm[0,1] if len(cm)>1 else 0, '假阴性': cm[1,0] if len(cm)>1 else 0
        })
        model_predictions[name] = y_pred
        model_probabilities[name] = y_prob
        
        print(f"   ✔ F1分数: {f1:.4f}，训练时间: {time.time()-start_time:.2f}秒")
        
    except Exception as e:
        print(f"   ✗ 训练失败: {str(e)}")
        continue

del model_objects
gc.collect()
print("   ✔ 已删除所有模型对象，释放大量内存")

if not results:
    print("\n   ✗ 没有可用的模型结果，无法生成对比图表")
    print("\n" + "="*80)
    print("模型对比分析完成！")
    print(f"生成图表路径: {current_dir}")
    print("="*80)
    exit()

results_df = pd.DataFrame(results).sort_values('F1分数', ascending=False)
best_model = results_df.iloc[0]
best_model_name = best_model['模型']

blue_colors = ['#1E88E5', '#64B5F6', '#90CAF9', '#BBDEFB']
radar_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
radar_model_colors = dict(zip(results_df['模型'], radar_colors[:len(results_df)]))


# ---------------------- 模型集成分析----------------------
print("\n10.10 模型集成分析")
print("-" * 40)

ensemble_models = []
for name in results_df['模型']:
    if name == '逻辑回归':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif name == '随机森林':
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    elif name == '决策树':
        model = DecisionTreeClassifier(random_state=42, max_depth=10)
    else:
        model = GaussianNB()
    ensemble_models.append((name, model))

# 软投票集成
soft_voter = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'
)
soft_voter.fit(X_train_scaled, y_train)
y_pred_soft = soft_voter.predict(X_test_scaled)

# 硬投票集成
hard_voter = VotingClassifier(
    estimators=ensemble_models,
    voting='hard'
)
hard_voter.fit(X_train_scaled, y_train)
y_pred_hard = hard_voter.predict(X_test_scaled)

# 评估集成模型
def evaluate_voter(y_pred, name):
    print(f"\n{name}集成性能:")
    print(f"  准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  精确率: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  召回率: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1分数: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    return f1_score(y_test, y_pred, zero_division=0)

soft_f1 = evaluate_voter(y_pred_soft, "软投票")
hard_f1 = evaluate_voter(y_pred_hard, "硬投票")

# 对比最佳单模型
best_single_f1 = best_model['F1分数']
soft_improve = (soft_f1 - best_single_f1) / best_single_f1 * 100 if best_single_f1 != 0 else 0
hard_improve = (hard_f1 - best_single_f1) / best_single_f1 * 100 if best_single_f1 != 0 else 0

print("\n【集成效果对比】")
print(f"最佳单模型F1分数: {best_single_f1:.4f}")
print(f"软投票提升: {soft_improve:.2f}%")
print(f"硬投票提升: {hard_improve:.2f}%")

del X_train_scaled, X_test_scaled, ensemble_models, soft_voter, hard_voter
del model_predictions, model_probabilities, results_df, best_model
gc.collect()
print("\n" + "="*80)
print("模型对比分析完成！")
print(f"生成图表路径: {current_dir}")
print("="*80)

