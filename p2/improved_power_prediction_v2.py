# 改进版光伏发电功率预测模型 - 使用真实历史数据 + 可视化
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# 更兼容的中文字体设置
import matplotlib as mpl
try:
    # 尝试多种中文字体
    font_candidates = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc', 
        'C:/Windows/Fonts/msyh.ttc',
        'SimHei', 'Microsoft YaHei', 'SimSun'
    ]
    
    font_set = False
    for font in font_candidates:
        try:
            if font.endswith('.ttf') or font.endswith('.ttc'):
                mpl.font_manager.fontManager.addfont(font)
                font_name = mpl.font_manager.FontProperties(fname=font).get_name()
                plt.rcParams['font.sans-serif'] = [font_name]
            else:
                plt.rcParams['font.sans-serif'] = [font]
            font_set = True
            break
        except:
            continue
    
    if not font_set:
        # 如果都失败，使用默认设置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
except Exception as e:
    # 备用方案
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 设置不显示图片弹窗
plt.ioff()

class ImprovedPowerPredictionV2:
    """改进版光伏发电功率预测模型 - 使用真实历史数据"""
    
    def __init__(self, station_id: str):
        self.station_id = station_id
        self.model = None
        self.feature_names = []
        self.historical_patterns = {}
        self.capacity = None
        self.train_data = None
        self.test_data = None
        self.predictions = None
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print(f"📊 加载 {self.station_id} 数据...")
        
        # 加载数据
        df = pd.read_csv(f'data/{self.station_id}.csv')
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        
        # 估算开机容量（历史最大功率的1.1倍）
        self.capacity = df['power'].max() * 1.1
        
        print(f"  数据时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
        print(f"  数据点数: {len(df):,}")
        print(f"  平均功率: {df['power'].mean():.3f} MW")
        print(f"  最大功率: {df['power'].max():.3f} MW")
        print(f"  估算开机容量: {self.capacity:.3f} MW")
        
        return df
    
    def create_features(self, df):
        """创建特征工程"""
        print("🔧 创建特征工程...")
        
        # 复制数据
        data = df.copy()
        
        # 时间特征
        data['hour'] = data['date_time'].dt.hour
        data['minute'] = data['date_time'].dt.minute
        data['day_of_week'] = data['date_time'].dt.dayofweek
        data['day_of_year'] = data['date_time'].dt.dayofyear
        data['month'] = data['date_time'].dt.month
        data['time_slot'] = data['hour'] * 4 + data['minute'] // 15
        
        # 周期性特征
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['time_slot_sin'] = np.sin(2 * np.pi * data['time_slot'] / 96)
        data['time_slot_cos'] = np.cos(2 * np.pi * data['time_slot'] / 96)
        
        # 滞后特征
        lag_periods = [1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672]
        for lag in lag_periods:
            data[f'power_lag_{lag}'] = data['power'].shift(lag)
        
        # 滚动统计特征
        windows = [4, 8, 12, 24, 48, 96]
        for window in windows:
            data[f'power_rolling_mean_{window}'] = data['power'].rolling(window=window).mean()
            data[f'power_rolling_std_{window}'] = data['power'].rolling(window=window).std()
            data[f'power_rolling_max_{window}'] = data['power'].rolling(window=window).max()
            data[f'power_rolling_min_{window}'] = data['power'].rolling(window=window).min()
        
        # 差分特征
        data['power_diff_1'] = data['power'].diff(1)
        data['power_diff_4'] = data['power'].diff(4)
        data['power_diff_24'] = data['power'].diff(24)
        data['power_diff_96'] = data['power'].diff(96)
        
        # 历史同期特征
        data['power_same_hour_7d'] = data['power'].shift(7*96)
        data['power_same_hour_14d'] = data['power'].shift(14*96)
        data['power_same_hour_30d'] = data['power'].shift(30*96)
        
        # 趋势特征（简化版，避免数值问题）
        data['power_trend_short'] = data['power'].rolling(window=12).mean().diff()
        data['power_trend_long'] = data['power'].rolling(window=96).mean().diff()
        
        # 白天判断（基于UTC时间，22:00-10:00为白天）
        data['is_daytime'] = ((data['hour'] >= 22) | (data['hour'] <= 10)).astype(int)
        
        # 功率变化率（处理除零问题）
        data['power_change_rate'] = data['power'].pct_change().fillna(0)
        data['power_change_rate'] = data['power_change_rate'].replace([np.inf, -np.inf], 0)
        data['power_acceleration'] = data['power_change_rate'].diff().fillna(0)
        
        # 填充缺失值和处理无穷大值
        data = data.fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
        
        # 检查并处理异常值
        for col in data.columns:
            if col not in ['date_time']:
                # 将极大值限制在合理范围内
                if data[col].dtype in ['float64', 'int64']:
                    q99 = data[col].quantile(0.99)
                    q01 = data[col].quantile(0.01)
                    data[col] = data[col].clip(lower=q01, upper=q99)
        
        print(f"  特征数量: {len(data.columns) - 2}")  # 减去date_time和power
        
        return data
    
    def split_data(self, data, test_days=7):
        """分割训练和测试数据"""
        print(f"📊 分割数据 - 最后{test_days}天作为测试集...")
        
        # 按时间分割，最后test_days天作为测试集
        test_start_idx = len(data) - test_days * 96  # 每天96个15分钟间隔
        
        train_data = data[:test_start_idx].copy()
        test_data = data[test_start_idx:].copy()
        
        print(f"  训练集: {len(train_data):,} 条记录")
        print(f"  测试集: {len(test_data):,} 条记录")
        print(f"  训练集时间范围: {train_data['date_time'].min()} 到 {train_data['date_time'].max()}")
        print(f"  测试集时间范围: {test_data['date_time'].min()} 到 {test_data['date_time'].max()}")
        
        return train_data, test_data
    
    def train_model(self, train_data):
        """训练XGBoost模型"""
        print("🚀 训练XGBoost模型...")
        
        # 只使用白天时段的数据进行训练
        daytime_mask = train_data['is_daytime'] == 1
        train_subset = train_data[daytime_mask].copy()
        
        # 移除缺失值过多的行
        train_subset = train_subset.dropna()
        
        if len(train_subset) == 0:
            raise ValueError("没有足够的训练数据")
        
        # 特征和目标变量
        feature_cols = [col for col in train_subset.columns 
                       if col not in ['date_time', 'power']]
        X_train = train_subset[feature_cols]
        y_train = train_subset['power']
        
        self.feature_names = feature_cols
        
        # 训练模型
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=10,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 训练集性能
        train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        
        print(f"  训练集 R²: {train_r2:.4f}")
        print(f"  训练集 MAE: {train_mae:.4f}")
        
        return self.model
    
    def predict_test_period(self, test_data):
        """预测测试期间的功率"""
        print("🔮 预测测试期间功率...")
        
        predictions = []
        
        for idx, row in test_data.iterrows():
            # 准备特征
            features = row[self.feature_names].values.reshape(1, -1)
            
            # 预测
            pred = self.model.predict(features)[0]
            
            # 确保预测值非负
            pred = max(0, pred)
            
            # 夜间时段设为0
            if row['is_daytime'] == 0:
                pred = 0
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def calculate_evaluation_metrics(self, actual, predicted):
        """计算评价指标"""
        # 只计算白天时段的指标
        daytime_mask = (actual > 0) | (predicted > 0)
        
        if np.sum(daytime_mask) == 0:
            return {}
        
        actual_day = actual[daytime_mask]
        predicted_day = predicted[daytime_mask]
        
        # 归一化误差（相对于开机容量）
        normalized_actual = actual_day / self.capacity
        normalized_predicted = predicted_day / self.capacity
        normalized_error = normalized_predicted - normalized_actual
        
        # 计算各项指标
        rmse = np.sqrt(np.mean(normalized_error ** 2))
        mae = np.mean(np.abs(normalized_error))
        me = np.mean(normalized_error)
        
        # 相关系数
        if len(actual_day) > 1:
            correlation = np.corrcoef(actual_day, predicted_day)[0, 1]
        else:
            correlation = 0
        
        # 准确率
        accuracy = (1 - rmse) * 100
        
        # 合格率（误差小于25%的比例）
        qualification_mask = np.abs(normalized_error) < 0.25
        qualification_rate = np.mean(qualification_mask) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'ME': me,
            'Correlation': correlation,
            'Accuracy': accuracy,
            'Qualification_Rate': qualification_rate,
            'Sample_Count': len(actual_day)
        }
    
    def create_visualizations(self, test_data, predictions):
        """创建可视化图表"""
        print("📊 生成可视化图表...")
        
        # 创建图表目录
        fig_dir = Path("results/figures")
        fig_dir.mkdir(exist_ok=True)
        
        # 1. 预测vs实际对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.station_id} 光伏发电功率预测分析', fontsize=16, fontweight='bold')
        
        # 时间序列对比
        ax1 = axes[0, 0]
        time_range = test_data['date_time']
        ax1.plot(time_range, test_data['power'], label='实际功率', linewidth=2, alpha=0.8)
        ax1.plot(time_range, predictions, label='预测功率', linewidth=2, alpha=0.8)
        ax1.set_title('预测vs实际功率时间序列对比')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('功率 (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax1.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 散点图
        ax2 = axes[0, 1]
        ax2.scatter(test_data['power'], predictions, alpha=0.6, s=20)
        max_power = max(test_data['power'].max(), predictions.max())
        ax2.plot([0, max_power], [0, max_power], 'r--', label='理想预测线')
        ax2.set_title('预测vs实际功率散点图')
        ax2.set_xlabel('实际功率 (MW)')
        ax2.set_ylabel('预测功率 (MW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 误差分布
        ax3 = axes[1, 0]
        errors = predictions - test_data['power']
        ax3.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(errors.mean(), color='red', linestyle='--', 
                   label=f'平均误差: {errors.mean():.3f}')
        ax3.set_title('预测误差分布')
        ax3.set_xlabel('预测误差 (MW)')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 日内功率模式
        ax4 = axes[1, 1]
        test_data_copy = test_data.copy()
        test_data_copy['hour'] = test_data_copy['date_time'].dt.hour
        test_data_copy['predictions'] = predictions
        
        hourly_actual = test_data_copy.groupby('hour')['power'].mean()
        hourly_predicted = test_data_copy.groupby('hour')['predictions'].mean()
        
        ax4.plot(hourly_actual.index, hourly_actual.values, 'o-', label='实际平均功率', linewidth=2)
        ax4.plot(hourly_predicted.index, hourly_predicted.values, 's-', label='预测平均功率', linewidth=2)
        ax4.set_title('日内平均功率模式对比')
        ax4.set_xlabel('小时')
        ax4.set_ylabel('平均功率 (MW)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{self.station_id}_prediction_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图片，不显示弹窗
        
        # 2. 特征重要性图
        if hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance(fig_dir)
        
        # 3. 评价指标雷达图
        metrics = self.calculate_evaluation_metrics(test_data['power'].values, predictions)
        if metrics:
            self.plot_metrics_radar(metrics, fig_dir)
        
        # 4. 每日预测性能对比
        self.plot_daily_performance(test_data, predictions, fig_dir)
    
    def plot_feature_importance(self, fig_dir):
        """绘制特征重要性图"""
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 取前20个重要特征
        top_features = feature_importance_df.head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'{self.station_id} 特征重要性排序 (Top 20)')
        plt.xlabel('重要性得分')
        plt.ylabel('特征名称')
        plt.tight_layout()
        plt.savefig(fig_dir / f'{self.station_id}_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图片，不显示弹窗
    
    def plot_metrics_radar(self, metrics, fig_dir):
        """绘制评价指标雷达图"""
        # 准备雷达图数据
        categories = ['准确率', '合格率', '相关系数', 'RMSE(反)', 'MAE(反)', 'ME(反)']
        values = [
            metrics['Accuracy'],
            metrics['Qualification_Rate'],
            metrics['Correlation'] * 100,  # 转换为百分比
            (1 - metrics['RMSE']) * 100,   # 反向，越大越好
            (1 - metrics['MAE']) * 100,    # 反向，越大越好
            (1 - abs(metrics['ME'])) * 100  # 反向，越大越好
        ]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values += values[:1]  # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=self.station_id)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(f'{self.station_id} 预测性能雷达图', size=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{self.station_id}_metrics_radar.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图片，不显示弹窗
    
    def plot_daily_performance(self, test_data, predictions, fig_dir):
        """绘制每日预测性能对比"""
        test_data_copy = test_data.copy()
        test_data_copy['predictions'] = predictions
        test_data_copy['date'] = test_data_copy['date_time'].dt.date
        
        # 计算每日统计
        daily_stats = test_data_copy.groupby('date').agg({
            'power': ['mean', 'max', 'sum'],
            'predictions': ['mean', 'max', 'sum']
        }).round(3)
        
        daily_stats.columns = ['实际平均', '实际最大', '实际总和', '预测平均', '预测最大', '预测总和']
        
        # 绘制每日对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.station_id} 每日预测性能对比', fontsize=16, fontweight='bold')
        
        # 每日平均功率
        ax1 = axes[0, 0]
        ax1.plot(daily_stats.index, daily_stats['实际平均'], 'o-', label='实际平均', linewidth=2)
        ax1.plot(daily_stats.index, daily_stats['预测平均'], 's-', label='预测平均', linewidth=2)
        ax1.set_title('每日平均功率对比')
        ax1.set_ylabel('平均功率 (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 每日最大功率
        ax2 = axes[0, 1]
        ax2.plot(daily_stats.index, daily_stats['实际最大'], 'o-', label='实际最大', linewidth=2)
        ax2.plot(daily_stats.index, daily_stats['预测最大'], 's-', label='预测最大', linewidth=2)
        ax2.set_title('每日最大功率对比')
        ax2.set_ylabel('最大功率 (MW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 每日发电量
        ax3 = axes[1, 0]
        ax3.plot(daily_stats.index, daily_stats['实际总和'], 'o-', label='实际发电量', linewidth=2)
        ax3.plot(daily_stats.index, daily_stats['预测总和'], 's-', label='预测发电量', linewidth=2)
        ax3.set_title('每日发电量对比')
        ax3.set_ylabel('发电量 (MWh)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 每日误差
        ax4 = axes[1, 1]
        daily_error = daily_stats['预测平均'] - daily_stats['实际平均']
        ax4.bar(range(len(daily_error)), daily_error, alpha=0.7)
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_title('每日平均功率预测误差')
        ax4.set_ylabel('预测误差 (MW)')
        ax4.set_xticks(range(len(daily_error)))
        ax4.set_xticklabels([str(d) for d in daily_stats.index], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{self.station_id}_daily_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图片，不显示弹窗
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print(f"🎯 开始 {self.station_id} 完整分析...")
        
        # 1. 加载数据
        df = self.load_and_preprocess_data()
        
        # 2. 特征工程
        data = self.create_features(df)
        
        # 3. 分割数据
        self.train_data, self.test_data = self.split_data(data, test_days=7)
        
        # 4. 训练模型
        self.train_model(self.train_data)
        
        # 5. 预测
        self.predictions = self.predict_test_period(self.test_data)
        
        # 6. 评估
        metrics = self.calculate_evaluation_metrics(
            self.test_data['power'].values, 
            self.predictions
        )
        
        # 7. 可视化
        self.create_visualizations(self.test_data, self.predictions)
        
        # 8. 保存结果
        self.save_results(metrics)
        
        return metrics
    
    def save_results(self, metrics):
        """保存结果"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'date_time': self.test_data['date_time'],
            'actual_power': self.test_data['power'],
            'predicted_power': self.predictions,
            'error': self.predictions - self.test_data['power']
        })
        
        results_df.to_csv(results_dir / f'{self.station_id}_prediction_results.csv', index=False)
        
        # 保存模型
        joblib.dump(self.model, results_dir / f'{self.station_id}_xgboost_model.pkl')
        
        # 保存评价指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(results_dir / f'{self.station_id}_metrics.csv', index=False)
        
        print(f"✅ 结果已保存到 results/ 目录")

def run_multi_station_analysis():
    """运行多站点分析"""
    stations = ['station00', 'station04', 'station05', 'station09']
    all_metrics = {}
    
    print("🚀 开始多站点光伏发电功率预测分析...")
    print("="*60)
    
    for station in stations:
        try:
            print(f"\n{'='*20} {station} {'='*20}")
            predictor = ImprovedPowerPredictionV2(station)
            metrics = predictor.run_complete_analysis()
            all_metrics[station] = metrics
            
            print(f"\n📊 {station} 评价指标:")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  相关系数: {metrics['Correlation']:.4f}")
            print(f"  准确率: {metrics['Accuracy']:.2f}%")
            print(f"  合格率: {metrics['Qualification_Rate']:.2f}%")
            
        except Exception as e:
            print(f"❌ {station} 分析失败: {str(e)}")
            continue
    
    # 生成综合对比报告
    if all_metrics:
        create_summary_comparison(all_metrics)
    
    print(f"\n🎉 多站点分析完成！")
    return all_metrics

def create_summary_comparison(all_metrics):
    """创建综合对比报告"""
    print(f"\n{'='*60}")
    print("📊 多站点预测性能综合对比")
    print("="*60)
    
    # 创建对比表格
    comparison_data = []
    for station, metrics in all_metrics.items():
        comparison_data.append({
            '站点': station,
            'RMSE': f"{metrics['RMSE']:.6f}",
            'MAE': f"{metrics['MAE']:.6f}",
            '相关系数': f"{metrics['Correlation']:.4f}",
            '准确率(%)': f"{metrics['Accuracy']:.2f}",
            '合格率(%)': f"{metrics['Qualification_Rate']:.2f}",
            '样本数': metrics['Sample_Count']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 保存对比结果
    comparison_df.to_csv('results/multi_station_comparison.csv', index=False)
    
    # 创建综合对比可视化
    create_comparison_visualization(all_metrics)

def create_comparison_visualization(all_metrics):
    """创建综合对比可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多站点预测性能综合对比', fontsize=16, fontweight='bold')
    
    stations = list(all_metrics.keys())
    
    # RMSE对比
    ax1 = axes[0, 0]
    rmse_values = [all_metrics[s]['RMSE'] for s in stations]
    bars1 = ax1.bar(stations, rmse_values, alpha=0.7)
    ax1.set_title('RMSE对比 (越小越好)')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 准确率对比
    ax2 = axes[0, 1]
    accuracy_values = [all_metrics[s]['Accuracy'] for s in stations]
    bars2 = ax2.bar(stations, accuracy_values, alpha=0.7, color='green')
    ax2.set_title('准确率对比 (越大越好)')
    ax2.set_ylabel('准确率 (%)')
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, accuracy_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 合格率对比
    ax3 = axes[1, 0]
    qr_values = [all_metrics[s]['Qualification_Rate'] for s in stations]
    bars3 = ax3.bar(stations, qr_values, alpha=0.7, color='orange')
    ax3.set_title('合格率对比 (越大越好)')
    ax3.set_ylabel('合格率 (%)')
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, qr_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 相关系数对比
    ax4 = axes[1, 1]
    corr_values = [all_metrics[s]['Correlation'] for s in stations]
    bars4 = ax4.bar(stations, corr_values, alpha=0.7, color='purple')
    ax4.set_title('相关系数对比 (越大越好)')
    ax4.set_ylabel('相关系数')
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars4, corr_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/figures/multi_station_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图片，不显示弹窗

if __name__ == "__main__":
    # 运行多站点分析
    metrics = run_multi_station_analysis() 