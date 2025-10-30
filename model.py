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

# ---------------------- 3. 生成9张单独图表----------------------
DPI = 150

# 10.1 综合性能指标对比
plt.figure(figsize=(10, 6))
metrics = ['准确率', '精确率', '召回率', 'F1分数']
x = np.arange(len(results_df))
width = 0.2
for i, metric in enumerate(metrics):
    plt.bar(x + i*width - 0.3, results_df[metric], width, label=metric, color=blue_colors[i], alpha=0.8)
plt.xticks(x, results_df['模型'], rotation=15)
plt.xlabel('模型')
plt.ylabel('分数')
plt.title('10.1 各模型综合性能指标对比')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig(get_save_path('10.1_综合性能指标对比.png'), dpi=DPI, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.1_综合性能指标对比.png")

# 10.2 ROC曲线对比
plt.figure(figsize=(10, 6))
model_list = list(model_probabilities.keys())
for i, name in enumerate(model_list):
    fpr, tpr, _ = roc_curve(y_test, model_probabilities[name])
    auc_score = results_df[results_df['模型'] == name]['AUC'].values[0]
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})', 
             color=blue_colors[i], linewidth=2)
    del fpr, tpr
plt.plot([0,1], [0,1], 'k--', alpha=0.5)
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('10.2 各模型ROC曲线对比')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(get_save_path('10.2_ROC曲线对比.png'), dpi=DPI, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.2_ROC曲线对比.png")

# 10.3 最佳模型混淆矩阵
plt.figure(figsize=(8, 6))
best_pred = model_predictions[best_model_name]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['非严重', '严重'], yticklabels=['非严重', '严重'])
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.title(f'10.3 {best_model_name}混淆矩阵')
plt.text(0.5, -0.15, '注：对角线为正确预测，非对角线为错误预测', 
        ha='center', va='center', transform=plt.gca().transAxes)
plt.savefig(get_save_path('10.3_最佳模型混淆矩阵.png'), dpi=DPI, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.3_最佳模型混淆矩阵.png")

# 10.4 训练时间对比
plt.figure(figsize=(10, 6))
bars = plt.bar(results_df['模型'], results_df['训练时间(秒)'], 
              color=blue_colors[:len(results_df)], alpha=0.8)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, height, f'{height:.2f}s', ha='center', va='bottom')
plt.xlabel('模型')
plt.ylabel('训练时间 (秒)')
plt.title('10.4 各模型训练时间对比')
plt.grid(axis='y', alpha=0.3)
plt.savefig(get_save_path('10.4_训练时间对比.png'), dpi=DPI, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.4_训练时间对比.png")

# 10.5 精确率vs召回率权衡
plt.figure(figsize=(10, 6))
for i, row in enumerate(results_df.itertuples()):
    plt.scatter(row.召回率, row.精确率, s=100, color=blue_colors[i], label=row.模型, alpha=0.8)
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('10.5 精确率与召回率权衡关系')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(get_save_path('10.5_精确率vs召回率.png'), dpi=DPI, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.5_精确率vs召回率.png")

# 10.6 交叉验证分数分布
plt.figure(figsize=(10, 6))
cv_data = []
cv_labels = []
for name in models.keys():
    if name not in results_df['模型'].values:
        continue
    if name == '逻辑回归':
        temp_model = LogisticRegression(random_state=42, max_iter=1000)
    elif name == '随机森林':
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    elif name == '决策树':
        temp_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    else:
        temp_model = GaussianNB()
    scores = cross_val_score(temp_model, X_train_scaled, y_train, cv=5, scoring='f1')
    cv_data.append(scores)
    cv_labels.append(name)
    del temp_model, scores

bp = plt.boxplot(cv_data, labels=cv_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], blue_colors[:len(cv_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
plt.ylabel('F1分数')
plt.title('10.6 5折交叉验证分数分布')
plt.grid(axis='y', alpha=0.3)
plt.savefig(get_save_path('10.6_交叉验证分数分布.png'), dpi=DPI, bbox_inches='tight')
plt.close()
del cv_data, cv_labels
gc.collect()
print("   ✔ 生成：10.6_交叉验证分数分布.png")

# 10.7 特征重要性
plt.figure(figsize=(10, 6))
feature_imp = None

if '随机森林' in results_df['模型'].values:
    temp_rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=1)
    temp_rf.fit(X_train_scaled, y_train)
    importances = temp_rf.feature_importances_
    feature_imp = pd.DataFrame({'特征': available_features, '重要性': importances}).sort_values('重要性')
    plt.barh(feature_imp['特征'], feature_imp['重要性'], color=blue_colors[1], alpha=0.8)
    plt.title('10.7 随机森林特征重要性')
    del temp_rf
elif '决策树' in results_df['模型'].values:
    temp_dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    temp_dt.fit(X_train_scaled, y_train)
    importances = temp_dt.feature_importances_
    feature_imp = pd.DataFrame({'特征': available_features, '重要性': importances}).sort_values('重要性')
    plt.barh(feature_imp['特征'], feature_imp['重要性'], color=blue_colors[1], alpha=0.8)
    plt.title('10.7 决策树特征重要性')
    del temp_dt

if feature_imp is not None:
    plt.xlabel('重要性')

    for i, (_, row) in enumerate(feature_imp.iterrows()):
        plt.text(row['重要性'] + 0.005, i, f'{row["重要性"]:.3f}', ha='left', va='center')

    plt.savefig(get_save_path('10.7_特征重要性.png'), dpi=150, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.7_特征重要性.png")


# 10.8 最佳模型学习曲线
plt.figure(figsize=(10, 6))

if best_model_name == '逻辑回归':
    best_model_obj = LogisticRegression(random_state=42, max_iter=1000)
elif best_model_name == '随机森林':
    best_model_obj = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=1)
elif best_model_name == '决策树':
    best_model_obj = DecisionTreeClassifier(random_state=42, max_depth=10)
else:
    best_model_obj = GaussianNB()

# 计算学习曲线
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    best_model_obj, X_train_scaled, y_train, train_sizes=train_sizes, cv=5, n_jobs=1
)
del best_model_obj

# 计算均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)
del train_scores, val_scores

# 绘制学习曲线
plt.plot(train_sizes_abs, train_mean, 'o-', color=blue_colors[0], label='训练集', linewidth=2)
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                 alpha=0.2, color=blue_colors[0])
plt.plot(train_sizes_abs, val_mean, 's-', color=blue_colors[2], label='验证集', linewidth=2)
plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                 alpha=0.2, color=blue_colors[2])

plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.title(f'10.8 {best_model_name}学习曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(get_save_path('10.8_最佳模型学习曲线.png'), dpi=150, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.8_最佳模型学习曲线.png")


# 10.9 性能雷达图
plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='polar')
metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

# 绘制每个模型的雷达图
for _, row in enumerate(results_df.itertuples()):
    model_name = row.模型
    values = [row.准确率, row.精确率, row.召回率, row.F1分数, row.AUC]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=radar_model_colors[model_name])
    ax.fill(angles, values, alpha=0.15, color=radar_model_colors[model_name])

# 美化雷达图
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('10.9 模型性能雷达图（彩色区分）', y=1.08, fontsize=13)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

plt.savefig(get_save_path('10.9_性能雷达图.png'), dpi=150, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10.9_性能雷达图.png")


# ---------------------- 生成1张合并图表----------------------
plt.figure(figsize=(22, 18))
metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
x = np.arange(len(results_df))
width = 0.2

# 1. 综合性能指标
plt.subplot(3, 3, 1)
for i, metric in enumerate(metrics[:4]):
    plt.bar(x + i*width - 0.3, results_df[metric], width, label=metric, 
            color=blue_colors[i], alpha=0.8)
plt.xticks(x, results_df['模型'], rotation=15)
plt.title('1. 综合性能指标', fontsize=10)
plt.legend(fontsize=8)
plt.grid(axis='y', alpha=0.3)

# 2. ROC曲线
plt.subplot(3, 3, 2)
model_list = list(model_probabilities.keys())
for i, name in enumerate(model_list):
    fpr, tpr, _ = roc_curve(y_test, model_probabilities[name])
    auc_score = results_df[results_df['模型'] == name]['AUC'].values[0]
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})', 
             color=blue_colors[i], linewidth=1.5)
    del fpr, tpr
plt.plot([0,1], [0,1], 'k--', alpha=0.5)
plt.title('2. ROC曲线', fontsize=10)
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

# 3. 最佳模型混淆矩阵
plt.subplot(3, 3, 3)
best_pred = model_predictions[best_model_name]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'fontsize':8})
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title(f'3. {best_model_name}混淆矩阵', fontsize=10)
del cm

# 4. 训练时间对比
plt.subplot(3, 3, 4)
plt.bar(results_df['模型'], results_df['训练时间(秒)'], 
        color=blue_colors[:len(results_df)], alpha=0.8)
plt.xticks(rotation=15, fontsize=8)
plt.title('4. 训练时间对比', fontsize=10)
plt.grid(axis='y', alpha=0.3)

# 5. 精确率vs召回率
plt.subplot(3, 3, 5)
for i, row in enumerate(results_df.itertuples()):
    plt.scatter(row.召回率, row.精确率, s=50, color=blue_colors[i], label=row.模型)
plt.title('5. 精确率vs召回率', fontsize=10)
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

# 6. 交叉验证分布
plt.subplot(3, 3, 6)
cv_data = []
cv_labels = []
for name in results_df['模型']:
    if name == '逻辑回归':
        temp_model = LogisticRegression(random_state=42, max_iter=1000)
    elif name == '随机森林':
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    elif name == '决策树':
        temp_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    else:
        temp_model = GaussianNB()
    scores = cross_val_score(temp_model, X_train_scaled, y_train, cv=5, scoring='f1')
    cv_data.append(scores)
    cv_labels.append(name)
    del temp_model, scores

bp = plt.boxplot(cv_data, labels=cv_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], blue_colors[:len(cv_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
plt.xticks(rotation=15, fontsize=8)
plt.title('6. 交叉验证分数分布', fontsize=10)
plt.grid(axis='y', alpha=0.3)
del cv_data, cv_labels

# 7. 特征重要性
plt.subplot(3, 3, 7)
if feature_imp is not None:
    plt.barh(feature_imp['特征'], feature_imp['重要性'], 
            color=blue_colors[1], alpha=0.8)
    plt.yticks(fontsize=8)
plt.title('7. 特征重要性', fontsize=10)
plt.grid(axis='x', alpha=0.3)

# 8. 最佳模型学习曲线
plt.subplot(3, 3, 8)
plt.plot(train_sizes_abs, train_mean, 'o-', color=blue_colors[0], label='训练集', linewidth=1.5)
plt.plot(train_sizes_abs, val_mean, 's-', color=blue_colors[2], label='验证集', linewidth=1.5)
plt.xticks(fontsize=8)
plt.title(f'8. {best_model_name}学习曲线', fontsize=10)
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
del train_sizes_abs, train_mean, train_std, val_mean, val_std

# 9. 性能雷达图
plt.subplot(3, 3, 9, projection='polar')
for _, row in enumerate(results_df.itertuples()):
    model_name = row.模型
    values = [row.准确率, row.精确率, row.召回率, row.F1分数, row.AUC]
    values += values[:1]
    plt.plot(angles, values, 'o-', linewidth=1.5, label=model_name, color=radar_model_colors[model_name])
    plt.fill(angles, values, alpha=0.15, color=radar_model_colors[model_name])
plt.xticks(angles[:-1], metrics, fontsize=8)
plt.title('9. 性能雷达图', fontsize=10, y=1.08)
plt.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
combined_plot_path = get_save_path('10_模型对比综合图表.png')
plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
plt.close()
gc.collect()
print("   ✔ 生成：10_模型对比综合图表.png")


# ---------------------- 弹窗展示综合图表----------------------
print("\n正在展示10部分综合图表...（关闭窗口继续运行）")
try:
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111)
    img = plt.imread(combined_plot_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('10. 模型对比分析综合图表', fontsize=18, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)
except Exception as e:
    print(f"   ✗ 综合图表弹窗失败: {str(e)}")
    # 备选：用系统默认程序打开
    if os.name == 'nt':  # Windows
        os.startfile(combined_plot_path)
    else:  # Mac/Linux
        os.system(f'open "{combined_plot_path}"' if os.name == 'posix' else f'xdg-open "{combined_plot_path}"')
    print(f"   ✔ 已用系统图片查看器打开综合图表")
del img
gc.collect()


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

