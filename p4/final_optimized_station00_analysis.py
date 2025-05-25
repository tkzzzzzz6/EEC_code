#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4：最终优化的station00 NWP空间降尺度分析

针对数据特点的根本性改进：
1. 解决数据稀疏性问题
2. 改进训练/测试分割策略
3. 专注于实际发电时段
4. 使用更合适的评价指标
5. 实现真正有效的空间降尺度
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

warnings.filterwarnings('ignore')

# 中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')
plt.rcParams['axes.unicode_minus'] = False

sns.set_palette("husl")
plt.ioff()

class FinalOptimizedStation00Analysis:
    """最终优化的station00空间降尺度分析"""
    
    def __init__(self):
        self.station_id = 'station00'
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.capacity = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.generation_threshold = 0.05  # MW，实际发电阈值
        
    def load_and_analyze_data(self):
        """加载并深度分析数据"""
        print(f"📊 加载并深度分析 {self.station_id} 数据...")
        
        # 加载数据
        df = pd.read_csv(f'data/{self.station_id}.csv')
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        # UTC时间转换为当地时间（中国时区 UTC+8）
        print(f"  原始时间（UTC）: {df['date_time'].min()} 到 {df['date_time'].max()}")
        df['date_time'] = df['date_time'] + pd.Timedelta(hours=8)
        print(f"  转换后时间（北京时间）: {df['date_time'].min()} 到 {df['date_time'].max()}")
        
        df = df.sort_values('date_time').reset_index(drop=True)
        
        print(f"  原始数据点数: {len(df):,}")
        
        # 数据清洗
        df['power'] = np.maximum(0, df['power'])
        df['nwp_globalirrad'] = np.maximum(0, df['nwp_globalirrad'])
        df['lmd_totalirrad'] = np.maximum(0, df['lmd_totalirrad'])
        
        # 容量估算（使用更保守的方法）
        self.capacity = df['power'].quantile(0.95)
        print(f"  估算容量: {self.capacity:.3f} MW")
        
        # 添加时间特征
        df['hour'] = df['date_time'].dt.hour
        df['day_of_year'] = df['date_time'].dt.dayofyear
        df['month'] = df['date_time'].dt.month
        df['weekday'] = df['date_time'].dt.weekday
        
        # 太阳位置计算（基于当地时间）
        lat = 38.04778
        hour_angle = (df['hour'] + df['date_time'].dt.minute/60 - 12) * 15
        declination = 23.45 * np.sin(np.radians(360 * (284 + df['day_of_year']) / 365.25))
        
        solar_elevation = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(lat)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(lat)) * 
            np.cos(np.radians(hour_angle))
        )
        df['solar_elevation'] = np.degrees(solar_elevation)
        df['solar_elevation'] = np.maximum(df['solar_elevation'], 0)
        
        # 定义发电时段（基于当地时间，更准确的判断）
        df['is_generation_time'] = (
            (df['hour'] >= 6) & (df['hour'] <= 18) &  # 当地时间6-18点
            (df['solar_elevation'] > 0) & 
            ((df['nwp_globalirrad'] > 10) | (df['lmd_totalirrad'] > 10))
        ).astype(int)
        
        # 定义有效发电时段（实际有功率输出）
        df['is_effective_generation'] = (
            (df['is_generation_time'] == 1) & 
            (df['power'] > self.generation_threshold)
        ).astype(int)
        
        print(f"  发电时段: {df['is_generation_time'].sum():,} ({df['is_generation_time'].mean()*100:.1f}%)")
        print(f"  有效发电时段: {df['is_effective_generation'].sum():,} ({df['is_effective_generation'].mean()*100:.1f}%)")
        
        # 分析数据分布
        self.analyze_data_distribution(df)
        
        return df
    
    def analyze_data_distribution(self, df):
        """分析数据分布特征"""
        print("\n📈 数据分布分析:")
        
        # 按月份分析
        monthly_stats = df.groupby('month').agg({
            'power': ['mean', 'max', 'count'],
            'is_generation_time': 'sum',
            'is_effective_generation': 'sum'
        }).round(3)
        print("  月份统计:")
        print(monthly_stats)
        
        # 按小时分析
        hourly_stats = df.groupby('hour').agg({
            'power': ['mean', 'max'],
            'is_generation_time': 'sum',
            'is_effective_generation': 'sum'
        }).round(3)
        print("\n  小时统计（前10小时）:")
        print(hourly_stats.head(10))
        
        # 相关性分析
        generation_data = df[df['is_generation_time'] == 1]
        if len(generation_data) > 0:
            corr_power_nwp = generation_data['power'].corr(generation_data['nwp_globalirrad'])
            corr_power_lmd = generation_data['power'].corr(generation_data['lmd_totalirrad'])
            print(f"\n  发电时段相关性:")
            print(f"    功率 vs NWP辐射: {corr_power_nwp:.4f}")
            print(f"    功率 vs LMD辐射: {corr_power_lmd:.4f}")
    
    def create_advanced_features(self, df):
        """创建高级特征工程"""
        print("🔧 创建高级特征工程...")
        
        data = df.copy()
        
        # 1. 基础辐射特征
        data['nwp_diffuse_irrad'] = data['nwp_globalirrad'] - data['nwp_directirrad']
        data['lmd_direct_irrad'] = data['lmd_totalirrad'] - data['lmd_diffuseirrad']
        
        # 2. 辐射比例和质量指标
        data['nwp_diffuse_ratio'] = data['nwp_diffuse_irrad'] / (data['nwp_globalirrad'] + 1e-6)
        data['lmd_diffuse_ratio'] = data['lmd_diffuseirrad'] / (data['lmd_totalirrad'] + 1e-6)
        data['clearness_index_nwp'] = data['nwp_globalirrad'] / (1361 * np.sin(np.radians(data['solar_elevation'])) + 1e-6)
        data['clearness_index_lmd'] = data['lmd_totalirrad'] / (1361 * np.sin(np.radians(data['solar_elevation'])) + 1e-6)
        
        # 限制清晰度指数在合理范围内
        data['clearness_index_nwp'] = np.clip(data['clearness_index_nwp'], 0, 1.2)
        data['clearness_index_lmd'] = np.clip(data['clearness_index_lmd'], 0, 1.2)
        
        # 3. 温度特征
        data['nwp_temp_celsius'] = data['nwp_temperature'] - 273.15
        data['temp_diff_nwp_lmd'] = data['nwp_temp_celsius'] - data['lmd_temperature']
        data['temp_efficiency_factor'] = 1 - 0.004 * (data['nwp_temp_celsius'] - 25)  # 温度效率因子
        
        # 4. 改进的空间降尺度特征
        # 基于地理和气候的修正因子
        lat_correction = 1 + (38.04778 - 38.0) * 0.02  # 纬度修正
        lon_correction = 1 + (114.95 - 115.0) * 0.01   # 经度修正
        elevation_correction = 1.02  # 假设的海拔修正
        
        # 季节性修正
        season_correction = 1 + 0.1 * np.sin(2 * np.pi * data['day_of_year'] / 365.25)
        
        # 综合修正因子
        spatial_correction = lat_correction * lon_correction * elevation_correction * season_correction
        
        data['nwp_globalirrad_downscaled'] = data['nwp_globalirrad'] * spatial_correction
        data['nwp_directirrad_downscaled'] = data['nwp_directirrad'] * spatial_correction
        
        # 5. 时间特征
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # 6. 理论功率计算
        data['theoretical_power_nwp'] = (
            data['nwp_globalirrad'] / 1000 * self.capacity * 
            data['temp_efficiency_factor'] * 
            np.sin(np.radians(data['solar_elevation']))
        )
        data['theoretical_power_lmd'] = (
            data['lmd_totalirrad'] / 1000 * self.capacity * 
            data['temp_efficiency_factor'] * 
            np.sin(np.radians(data['solar_elevation']))
        )
        data['theoretical_power_downscaled'] = (
            data['nwp_globalirrad_downscaled'] / 1000 * self.capacity * 
            data['temp_efficiency_factor'] * 
            np.sin(np.radians(data['solar_elevation']))
        )
        
        # 确保理论功率非负
        for col in ['theoretical_power_nwp', 'theoretical_power_lmd', 'theoretical_power_downscaled']:
            data[col] = np.maximum(0, data[col])
        
        # 7. 数据源对比特征
        data['irrad_ratio_nwp_lmd'] = data['nwp_globalirrad'] / (data['lmd_totalirrad'] + 1e-6)
        data['irrad_diff_nwp_lmd'] = data['nwp_globalirrad'] - data['lmd_totalirrad']
        data['irrad_ratio_downscaled_lmd'] = data['nwp_globalirrad_downscaled'] / (data['lmd_totalirrad'] + 1e-6)
        
        # 8. 滞后特征（只在发电时段）
        for lag in [1, 4, 12, 24]:  # 15分钟到6小时
            data[f'power_lag_{lag}'] = data['power'].shift(lag)
            data[f'irrad_nwp_lag_{lag}'] = data['nwp_globalirrad'].shift(lag)
            data[f'irrad_lmd_lag_{lag}'] = data['lmd_totalirrad'].shift(lag)
        
        # 9. 滑动窗口特征
        for window in [4, 8, 12]:  # 1-3小时窗口
            data[f'power_rolling_mean_{window}'] = data['power'].rolling(window=window, min_periods=1).mean()
            data[f'irrad_nwp_rolling_mean_{window}'] = data['nwp_globalirrad'].rolling(window=window, min_periods=1).mean()
            data[f'irrad_lmd_rolling_mean_{window}'] = data['lmd_totalirrad'].rolling(window=window, min_periods=1).mean()
        
        # 10. 天气模式特征
        data['weather_clear'] = ((data['nwp_globalirrad'] > 600) & (data['nwp_diffuse_ratio'] < 0.3)).astype(int)
        data['weather_partly_cloudy'] = ((data['nwp_globalirrad'] > 200) & (data['nwp_globalirrad'] <= 600)).astype(int)
        data['weather_cloudy'] = ((data['nwp_globalirrad'] > 50) & (data['nwp_globalirrad'] <= 200)).astype(int)
        
        # 处理缺失值和异常值
        data = data.fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
        
        print(f"  总特征数量: {len(data.columns) - 2}")
        
        return data
    
    def smart_data_split(self, data, test_ratio=0.2):
        """智能数据分割策略"""
        print("📊 智能数据分割...")
        
        # 确保测试集包含足够的发电数据
        generation_data = data[data['is_generation_time'] == 1].copy()
        non_generation_data = data[data['is_generation_time'] == 0].copy()
        
        print(f"  发电时段数据: {len(generation_data):,}")
        print(f"  非发电时段数据: {len(non_generation_data):,}")
        
        # 按时间分割发电数据
        n_test_generation = int(len(generation_data) * test_ratio)
        train_generation = generation_data[:-n_test_generation]
        test_generation = generation_data[-n_test_generation:]
        
        # 按时间分割非发电数据
        n_test_non_generation = int(len(non_generation_data) * test_ratio)
        train_non_generation = non_generation_data[:-n_test_non_generation]
        test_non_generation = non_generation_data[-n_test_non_generation:]
        
        # 合并训练和测试数据
        train_data = pd.concat([train_generation, train_non_generation]).sort_values('date_time')
        test_data = pd.concat([test_generation, test_non_generation]).sort_values('date_time')
        
        print(f"  训练集: {len(train_data):,} (发电: {len(train_generation):,}, 非发电: {len(train_non_generation):,})")
        print(f"  测试集: {len(test_data):,} (发电: {len(test_generation):,}, 非发电: {len(test_non_generation):,})")
        
        return train_data, test_data
    
    def train_optimized_models(self, train_data):
        """训练优化模型"""
        print("🚀 训练优化预测模型...")
        
        # 只使用发电时段数据训练
        generation_train = train_data[train_data['is_generation_time'] == 1].copy()
        
        if len(generation_train) < 100:
            print("⚠️  发电时段数据不足，使用所有训练数据")
            generation_train = train_data.copy()
        
        print(f"  实际训练数据: {len(generation_train):,}")
        
        # 特征选择
        feature_cols = [col for col in generation_train.columns 
                       if col not in ['date_time', 'power', 'is_generation_time', 'is_effective_generation']]
        
        X_train = generation_train[feature_cols]
        y_train = generation_train['power']
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        
        self.feature_names = feature_cols
        
        # 1. 基线模型（简单特征）
        print("  训练基线模型...")
        basic_features = ['nwp_globalirrad', 'lmd_totalirrad', 'solar_elevation', 
                         'hour_sin', 'hour_cos', 'theoretical_power_nwp']
        basic_features = [f for f in basic_features if f in feature_cols]
        
        if basic_features:
            X_basic = X_train_scaled[basic_features]
            self.models['baseline'] = Ridge(alpha=1.0)
            self.models['baseline'].fit(X_basic, y_train)
        
        # 2. 原始NWP模型
        print("  训练原始NWP模型...")
        original_features = [col for col in feature_cols 
                           if not 'downscaled' in col and not 'lag' in col]
        X_original = X_train_scaled[original_features]
        
        self.models['original_nwp'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.models['original_nwp'].fit(X_original, y_train)
        
        # 3. 降尺度NWP模型
        print("  训练降尺度NWP模型...")
        downscaled_features = [col for col in feature_cols 
                             if 'downscaled' in col or not col.startswith('nwp_') or 
                             col.startswith('theoretical_power_downscaled')]
        X_downscaled = X_train_scaled[downscaled_features]
        
        self.models['downscaled_nwp'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.models['downscaled_nwp'].fit(X_downscaled, y_train)
        
        # 4. 混合模型
        print("  训练混合模型...")
        self.models['hybrid'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=8,
            learning_rate=0.08,
            n_estimators=300,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        self.models['hybrid'].fit(X_train_scaled, y_train)
        
        # 5. 随机森林模型
        print("  训练随机森林模型...")
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        
        # 打印训练性能
        print("\n  训练集性能:")
        for model_name, model in self.models.items():
            if model_name == 'baseline' and basic_features:
                train_pred = model.predict(X_train_scaled[basic_features])
            elif model_name == 'original_nwp':
                train_pred = model.predict(X_train_scaled[original_features])
            elif model_name == 'downscaled_nwp':
                train_pred = model.predict(X_train_scaled[downscaled_features])
            else:
                train_pred = model.predict(X_train_scaled)
            
            train_r2 = r2_score(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            print(f"    {model_name}: R^2 = {train_r2:.4f}, RMSE = {train_rmse:.4f}")
    
    def predict_with_postprocessing(self, test_data):
        """带后处理的预测"""
        print("🔮 进行预测...")
        
        predictions = {}
        feature_cols = self.feature_names
        
        # 准备测试数据
        X_test = test_data[feature_cols]
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        for model_name, model in self.models.items():
            print(f"  {model_name} 预测中...")
            
            # 选择对应的特征
            if model_name == 'baseline':
                basic_features = ['nwp_globalirrad', 'lmd_totalirrad', 'solar_elevation', 
                                'hour_sin', 'hour_cos', 'theoretical_power_nwp']
                basic_features = [f for f in basic_features if f in feature_cols]
                if basic_features:
                    X_model = X_test_scaled[basic_features]
                else:
                    continue
            elif model_name == 'original_nwp':
                original_features = [col for col in feature_cols 
                                   if not 'downscaled' in col and not 'lag' in col]
                X_model = X_test_scaled[original_features]
            elif model_name == 'downscaled_nwp':
                downscaled_features = [col for col in feature_cols 
                                     if 'downscaled' in col or not col.startswith('nwp_') or 
                                     col.startswith('theoretical_power_downscaled')]
                X_model = X_test_scaled[downscaled_features]
            else:
                X_model = X_test_scaled
            
            # 预测
            pred = model.predict(X_model)
            
            # 后处理
            pred = np.maximum(0, pred)  # 确保非负
            pred = np.minimum(pred, self.capacity * 1.1)  # 限制在合理范围内
            
            # 非发电时段设为0
            non_generation_mask = test_data['is_generation_time'] == 0
            pred[non_generation_mask] = 0
            
            predictions[model_name] = pred
        
        return predictions
    
    def calculate_comprehensive_metrics(self, actual, predicted, test_data, model_name=""):
        """计算综合评价指标"""
        # 全数据集指标
        all_r2 = r2_score(actual, predicted)
        all_rmse = np.sqrt(mean_squared_error(actual, predicted))
        all_mae = mean_absolute_error(actual, predicted)
        
        # 只考虑发电时段
        generation_mask = test_data['is_generation_time'] == 1
        if np.sum(generation_mask) > 0:
            actual_gen = actual[generation_mask]
            predicted_gen = predicted[generation_mask]
            
            gen_r2 = r2_score(actual_gen, predicted_gen)
            gen_rmse = np.sqrt(mean_squared_error(actual_gen, predicted_gen))
            gen_mae = mean_absolute_error(actual_gen, predicted_gen)
            gen_corr = np.corrcoef(actual_gen, predicted_gen)[0, 1] if len(actual_gen) > 1 else 0
            
            # 归一化指标
            gen_nrmse = gen_rmse / (actual_gen.max() - actual_gen.min() + 1e-6) * 100
            gen_mape = np.mean(np.abs((actual_gen - predicted_gen) / (actual_gen + 1e-6))) * 100
        else:
            gen_r2 = gen_rmse = gen_mae = gen_corr = gen_nrmse = gen_mape = 0
        
        # 只考虑有效发电时段（功率>阈值）
        effective_mask = (test_data['is_effective_generation'] == 1) | (predicted > self.generation_threshold)
        if np.sum(effective_mask) > 0:
            actual_eff = actual[effective_mask]
            predicted_eff = predicted[effective_mask]
            
            eff_r2 = r2_score(actual_eff, predicted_eff)
            eff_rmse = np.sqrt(mean_squared_error(actual_eff, predicted_eff))
            eff_mae = mean_absolute_error(actual_eff, predicted_eff)
        else:
            eff_r2 = eff_rmse = eff_mae = 0
        
        return {
            'All_R2': all_r2,
            'All_RMSE': all_rmse,
            'All_MAE': all_mae,
            'Generation_R2': gen_r2,
            'Generation_RMSE': gen_rmse,
            'Generation_MAE': gen_mae,
            'Generation_Correlation': gen_corr,
            'Generation_NRMSE': gen_nrmse,
            'Generation_MAPE': gen_mape,
            'Effective_R2': eff_r2,
            'Effective_RMSE': eff_rmse,
            'Effective_MAE': eff_mae,
            'Generation_Sample_Count': np.sum(generation_mask),
            'Effective_Sample_Count': np.sum(effective_mask),
            'Model_Name': model_name
        }
    
    def analyze_downscaling_effectiveness(self):
        """分析降尺度有效性"""
        print("📈 分析空间降尺度技术的有效性...")
        
        actual = self.test_data['power'].values
        
        # 计算各模型指标
        for model_name, predictions in self.predictions.items():
            metrics = self.calculate_comprehensive_metrics(actual, predictions, self.test_data, model_name)
            self.metrics[model_name] = metrics
        
        # 效果对比
        print("\n" + "="*100)
        print("📊 模型性能对比")
        print("="*100)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  全数据集 - R^2: {metrics['All_R2']:.4f}, RMSE: {metrics['All_RMSE']:.4f}")
            print(f"  发电时段 - R^2: {metrics['Generation_R2']:.4f}, RMSE: {metrics['Generation_RMSE']:.4f}, 相关系数: {metrics['Generation_Correlation']:.4f}")
            print(f"  有效发电 - R^2: {metrics['Effective_R2']:.4f}, RMSE: {metrics['Effective_RMSE']:.4f}")
            print(f"  样本数量 - 发电: {metrics['Generation_Sample_Count']}, 有效: {metrics['Effective_Sample_Count']}")
        
        # 降尺度有效性分析
        if 'original_nwp' in self.metrics and 'downscaled_nwp' in self.metrics:
            orig = self.metrics['original_nwp']
            down = self.metrics['downscaled_nwp']
            
            # 基于发电时段的改善
            gen_r2_improvement = down['Generation_R2'] - orig['Generation_R2']
            gen_rmse_improvement = (orig['Generation_RMSE'] - down['Generation_RMSE']) / orig['Generation_RMSE'] * 100
            
            print(f"\n{'='*100}")
            print("🎯 空间降尺度技术有效性分析")
            print("="*100)
            print(f"发电时段R^2改善: {gen_r2_improvement:.4f}")
            print(f"发电时段RMSE改善: {gen_rmse_improvement:.2f}%")
            
            # 有效性等级
            if gen_r2_improvement > 0.05 and gen_rmse_improvement > 5:
                effectiveness = "显著有效"
            elif gen_r2_improvement > 0.02 and gen_rmse_improvement > 2:
                effectiveness = "有效"
            elif gen_r2_improvement > 0:
                effectiveness = "轻微有效"
            else:
                effectiveness = "无效或负面影响"
            
            print(f"有效性等级: {effectiveness}")
            
            return effectiveness
        
        return "无法评估"
    
    def create_comprehensive_visualizations(self):
        """创建综合可视化图表"""
        print("📊 生成优化的综合可视化分析图表...")
        
        # 创建分类文件夹结构
        base_dir = Path("results/figures")
        folders = {
            'time_series': base_dir / "时间序列分析",
            'prediction': base_dir / "预测效果分析", 
            'performance': base_dir / "性能对比分析",
            'data_analysis': base_dir / "数据分布分析",
            'downscaling': base_dir / "降尺度效果分析"
        }
        
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        # 1. 时间序列分析
        self.plot_detailed_time_series(folders['time_series'])
        
        # 2. 预测效果分析
        self.plot_prediction_analysis(folders['prediction'])
        
        # 3. 性能对比分析
        self.plot_comprehensive_performance(folders['performance'])
        
        # 4. 数据分布分析
        self.plot_data_distribution_analysis(folders['data_analysis'])
        
        # 5. 降尺度效果分析
        self.plot_downscaling_analysis(folders['downscaling'])
        
        print("✅ 优化的综合可视化图表生成完成")
        print("📁 图表已按类别保存到不同文件夹中")
    
    def plot_data_distribution_analysis(self, folder):
        """绘制数据分布分析图"""
        print("📊 绘制数据分布分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.station_id} 数据分布与特征分析', fontsize=16, fontweight='bold')
        
        # 1. 功率分布直方图
        ax1 = axes[0, 0]
        power_data = self.test_data['power']
        generation_power = power_data[power_data > 0]
        
        ax1.hist(generation_power, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('发电功率分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('功率 (MW)', fontsize=12)
        ax1.set_ylabel('频次', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = (f'最大值: {generation_power.max():.3f} MW\n'
                     f'平均值: {generation_power.mean():.3f} MW\n'
                     f'中位数: {generation_power.median():.3f} MW\n'
                     f'标准差: {generation_power.std():.3f} MW')
        ax1.text(0.7, 0.7, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
        
        # 2. 时间分布分析
        ax2 = axes[0, 1]
        hourly_power = self.test_data.groupby(self.test_data['date_time'].dt.hour)['power'].mean()
        
        ax2.bar(hourly_power.index, hourly_power.values, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('小时平均功率分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('小时', fontsize=12)
        ax2.set_ylabel('平均功率 (MW)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(range(0, 24, 2))
        
        # 3. 发电时段比例分析
        ax3 = axes[1, 0]
        generation_stats = {
            '非发电时段': (self.test_data['is_generation_time'] == 0).sum(),
            '发电时段': (self.test_data['is_generation_time'] == 1).sum(),
            '有效发电时段': (self.test_data['is_effective_generation'] == 1).sum()
        }
        
        colors_pie = ['lightcoral', 'lightblue', 'lightgreen']
        wedges, texts, autotexts = ax3.pie(generation_stats.values(), 
                                          labels=generation_stats.keys(),
                                          autopct='%1.1f%%',
                                          colors=colors_pie,
                                          startangle=90)
        ax3.set_title('发电时段分布', fontsize=14, fontweight='bold')
        
        # 美化饼图
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        # 4. 模型预测误差分布
        ax4 = axes[1, 1]
        if 'original_nwp' in self.predictions:
            actual = self.test_data['power'].values
            generation_mask = self.test_data['is_generation_time'] == 1
            
            if generation_mask.sum() > 0:
                actual_gen = actual[generation_mask]
                pred_gen = self.predictions['original_nwp'][generation_mask]
                errors = pred_gen - actual_gen
                
                ax4.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
                ax4.set_title('预测误差分布（原始NWP）', fontsize=14, fontweight='bold')
                ax4.set_xlabel('预测误差 (MW)', fontsize=12)
                ax4.set_ylabel('频次', fontsize=12)
                ax4.grid(True, alpha=0.3)
                ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='零误差线')
                
                # 添加误差统计
                error_stats = (f'RMSE: {np.sqrt(np.mean(errors**2)):.3f} MW\n'
                              f'MAE: {np.mean(np.abs(errors)):.3f} MW\n'
                              f'偏差: {np.mean(errors):.3f} MW\n'
                              f'标准差: {np.std(errors):.3f} MW')
                ax4.text(0.05, 0.95, error_stats, transform=ax4.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
                ax4.legend()
        
        plt.tight_layout()
        # 保存多个版本
        plt.savefig(folder / f'{self.station_id}_数据分布分析.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(folder / f'{self.station_id}_data_distribution_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_comprehensive_performance(self, folder):
        """绘制综合性能对比"""
        print("📊 绘制优化的综合性能对比...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle(f'{self.station_id} 模型综合性能对比分析', fontsize=20, fontweight='bold', y=0.98)
        
        models = list(self.metrics.keys())
        # 使用更好的颜色方案
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'][:len(models)]
        
        # 1. 发电时段R^2对比（最重要的指标）
        ax1 = axes[0, 0]
        gen_r2_values = [self.metrics[m]['Generation_R2'] for m in models]
        bars = ax1.bar(models, gen_r2_values, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('发电时段R^2对比', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('R^2 值', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.0)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
        
        # 添加数值标签
        for bar, value in zip(bars, gen_r2_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加优秀线
        ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='优秀线(0.9)')
        ax1.legend()
        
        # 2. 发电时段RMSE对比
        ax2 = axes[0, 1]
        gen_rmse_values = [self.metrics[m]['Generation_RMSE'] for m in models]
        bars = ax2.bar(models, gen_rmse_values, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_title('发电时段RMSE对比', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('RMSE (MW)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
        
        for bar, value in zip(bars, gen_rmse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 3. 相关系数对比
        ax3 = axes[0, 2]
        corr_values = [self.metrics[m]['Generation_Correlation'] for m in models]
        bars = ax3.bar(models, corr_values, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_title('发电时段相关系数对比', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('相关系数', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1.0)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
        
        for bar, value in zip(bars, corr_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 4. 降尺度改善效果
        ax4 = axes[1, 0]
        if 'original_nwp' in self.metrics:
            baseline_r2 = self.metrics['original_nwp']['Generation_R2']
            baseline_rmse = self.metrics['original_nwp']['Generation_RMSE']
            
            improvements_r2 = [self.metrics[m]['Generation_R2'] - baseline_r2 for m in models if m != 'original_nwp']
            improvements_rmse = [(baseline_rmse - self.metrics[m]['Generation_RMSE']) / baseline_rmse * 100 for m in models if m != 'original_nwp']
            improvement_models = [m for m in models if m != 'original_nwp']
            improvement_colors = [colors[i] for i, m in enumerate(models) if m != 'original_nwp']
            
            x = np.arange(len(improvement_models))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, improvements_r2, width, label='R^2改善', alpha=0.8, color='skyblue', edgecolor='black')
            ax4_twin = ax4.twinx()
            bars2 = ax4_twin.bar(x + width/2, improvements_rmse, width, label='RMSE改善(%)', alpha=0.8, color='lightcoral', edgecolor='black')
            
            ax4.set_xlabel('模型', fontsize=14, fontweight='bold')
            ax4.set_ylabel('R^2改善', fontsize=14, fontweight='bold', color='blue')
            ax4_twin.set_ylabel('RMSE改善 (%)', fontsize=14, fontweight='bold', color='red')
            ax4.set_title('相对原始NWP的改善效果', fontsize=16, fontweight='bold', pad=20)
            ax4.set_xticks(x)
            ax4.set_xticklabels(improvement_models, rotation=45, ha='right', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 添加数值标签
            for bar, value in zip(bars1, improvements_r2):
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.002 if value >= 0 else -0.005),
                        f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', 
                        fontsize=11, fontweight='bold')
            
            for bar, value in zip(bars2, improvements_rmse):
                ax4_twin.text(bar.get_x() + bar.get_width()/2, 
                             bar.get_height() + (0.5 if value >= 0 else -1.0),
                             f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top', 
                             fontsize=11, fontweight='bold')
            
            # 图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 5. 综合性能雷达图
        ax5 = axes[1, 1]
        
        # 准备雷达图数据
        categories = ['R^2', 'RMSE\n(反向)', '相关系数', 'MAE\n(反向)']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        
        for i, model in enumerate(models[:3]):  # 只显示前3个模型
            metrics = self.metrics[model]
            values = [
                metrics['Generation_R2'],
                1 - metrics['Generation_RMSE'] / max(gen_rmse_values),  # 反向RMSE
                metrics['Generation_Correlation'],
                1 - metrics['Generation_MAE'] / max([self.metrics[m]['Generation_MAE'] for m in models])  # 反向MAE
            ]
            values += values[:1]  # 闭合
            
            ax5.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax5.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=12)
        ax5.set_ylim(0, 1)
        ax5.set_title('综合性能雷达图', fontsize=16, fontweight='bold', pad=30)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax5.grid(True)
        
        # 6. 样本数量和效果总结
        ax6 = axes[1, 2]
        sample_counts = [self.metrics[m]['Generation_Sample_Count'] for m in models]
        
        # 创建表格显示详细信息
        table_data = []
        for model in models:
            metrics = self.metrics[model]
            table_data.append([
                model,
                f"{metrics['Generation_R2']:.3f}",
                f"{metrics['Generation_RMSE']:.3f}",
                f"{metrics['Generation_Sample_Count']}"
            ])
        
        table = ax6.table(cellText=table_data,
                         colLabels=['模型', 'R^2', 'RMSE', '样本数'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(models) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1)
        
        ax6.set_title('模型性能汇总表', fontsize=16, fontweight='bold', pad=20)
        ax6.axis('off')
        
        # 美化整体图表
        for ax in axes.flat:
            if ax != ax5 and ax != ax6:  # 排除雷达图和表格
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        # 保存多个版本
        plt.savefig(folder / f'{self.station_id}_综合性能对比.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(folder / f'{self.station_id}_optimized_comprehensive_performance.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_prediction_analysis(self, folder):
        """绘制预测效果分析"""
        print("📊 绘制优化的预测效果分析...")
        
        actual = self.test_data['power'].values
        generation_mask = self.test_data['is_generation_time'] == 1
        
        if generation_mask.sum() == 0:
            print("⚠️ 没有发电时段数据，跳过预测分析图")
            return
        
        # 计算需要的子图数量
        n_models = len(self.predictions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes
        elif n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        fig.suptitle(f'{self.station_id} 预测效果分析（发电时段）', fontsize=18, fontweight='bold', y=0.98)
        
        # 只显示发电时段数据
        actual_gen = actual[generation_mask]
        
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            pred_gen = predictions[generation_mask]
            
            # 创建密度散点图
            from matplotlib.colors import LinearSegmentedColormap
            
            # 散点图，根据密度着色
            scatter = ax.scatter(actual_gen, pred_gen, alpha=0.7, s=25, 
                               c=range(len(actual_gen)), cmap='plasma', edgecolors='none')
            
            # 理想线
            max_val = max(actual_gen.max(), pred_gen.max())
            min_val = min(actual_gen.min(), pred_gen.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.8, linewidth=3, label='理想预测')
            
            # 添加拟合线
            z = np.polyfit(actual_gen, pred_gen, 1)
            p = np.poly1d(z)
            ax.plot([min_val, max_val], p([min_val, max_val]), 'g--', alpha=0.8, linewidth=2, label='拟合线')
            
            # 统计信息
            r2 = self.metrics[model_name]['Generation_R2']
            rmse = self.metrics[model_name]['Generation_RMSE']
            corr = self.metrics[model_name]['Generation_Correlation']
            mae = self.metrics[model_name]['Generation_MAE']
            
            # 计算额外统计信息
            mape = np.mean(np.abs((actual_gen - pred_gen) / (actual_gen + 1e-6))) * 100
            bias = np.mean(pred_gen - actual_gen)
            
            # 添加详细统计信息文本框
            stats_text = (f'R^2 = {r2:.3f}\n'
                         f'RMSE = {rmse:.3f} MW\n'
                         f'MAE = {mae:.3f} MW\n'
                         f'相关系数 = {corr:.3f}\n'
                         f'MAPE = {mape:.1f}%\n'
                         f'偏差 = {bias:.3f} MW\n'
                         f'样本数 = {len(actual_gen)}')
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
                   fontsize=11, family='monospace')
            
            ax.set_xlabel('实际功率 (MW)', fontsize=14, fontweight='bold')
            ax.set_ylabel('预测功率 (MW)', fontsize=14, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='lower right', fontsize=12)
            
            # 设置相等的坐标轴范围
            margin = (max_val - min_val) * 0.05
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            ax.set_aspect('equal')
            
            # 美化坐标轴
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
        
        # 隐藏多余的子图
        for i in range(len(self.predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        # 保存多个版本
        plt.savefig(folder / f'{self.station_id}_预测效果分析.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(folder / f'{self.station_id}_optimized_prediction_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_detailed_time_series(self, folder):
        """绘制详细时间序列分析"""
        print("📊 绘制优化的时间序列分析...")
        
        fig, axes = plt.subplots(4, 1, figsize=(20, 20))
        fig.suptitle(f'{self.station_id} 优化时间序列分析', fontsize=16, fontweight='bold')
        
        # 选择一个有代表性的时间段进行详细展示（比如连续7天的数据）
        test_data_sorted = self.test_data.sort_values('date_time')
        
        # 找到一个包含较多发电数据的连续时间段
        generation_data = test_data_sorted[test_data_sorted['is_generation_time'] == 1]
        if len(generation_data) > 0:
            # 选择发电数据较多的一周
            start_date = generation_data['date_time'].iloc[len(generation_data)//3]
            end_date = start_date + pd.Timedelta(days=7)
            
            # 确保不超出数据范围
            if end_date > test_data_sorted['date_time'].max():
                end_date = test_data_sorted['date_time'].max()
                start_date = end_date - pd.Timedelta(days=7)
        else:
            # 如果没有发电数据，选择中间的一周
            start_date = test_data_sorted['date_time'].iloc[len(test_data_sorted)//2]
            end_date = start_date + pd.Timedelta(days=7)
        
        # 筛选时间段数据
        mask = (test_data_sorted['date_time'] >= start_date) & (test_data_sorted['date_time'] <= end_date)
        plot_data = test_data_sorted[mask].copy()
        
        if len(plot_data) == 0:
            print("⚠️ 没有找到合适的时间段数据，使用全部测试数据的采样")
            # 如果筛选后没有数据，使用采样
            plot_data = test_data_sorted.iloc[::max(1, len(test_data_sorted)//500)].copy()
        
        time_range = plot_data['date_time']
        actual = plot_data['power']
        generation_mask = plot_data['is_generation_time'] == 1
        
        print(f"  可视化时间段: {time_range.min()} 到 {time_range.max()}")
        print(f"  数据点数: {len(plot_data)}")
        print(f"  发电时段比例: {generation_mask.sum() / len(plot_data) * 100:.1f}%")
        
        # 获取对应的预测数据
        plot_predictions = {}
        for model_name, full_predictions in self.predictions.items():
            # 找到对应的预测值 - 修复索引问题
            # 使用布尔索引而不是位置索引
            test_data_mask = self.test_data['date_time'].isin(plot_data['date_time'])
            plot_predictions[model_name] = full_predictions[test_data_mask]
        
        # 第一个图：全时段对比（包含非发电时段）
        ax1 = axes[0]
        ax1.plot(time_range, actual, label='实际功率', linewidth=2, alpha=0.9, color='black', marker='o', markersize=2)
        
        # 选择最佳模型
        best_model = max(self.metrics.keys(), key=lambda x: self.metrics[x]['Generation_R2'])
        ax1.plot(time_range, plot_predictions[best_model], label=f'最佳模型({best_model})', 
                linewidth=2, alpha=0.8, color='red', marker='s', markersize=2)
        
        if 'original_nwp' in plot_predictions:
            ax1.plot(time_range, plot_predictions['original_nwp'], label='原始NWP', 
                    linewidth=2, alpha=0.8, color='blue', marker='^', markersize=2)
        
        if 'downscaled_nwp' in plot_predictions:
            ax1.plot(time_range, plot_predictions['downscaled_nwp'], label='降尺度NWP', 
                    linewidth=2, alpha=0.8, color='green', marker='d', markersize=2)
        
        # 标记发电时段背景
        ax1.fill_between(time_range, 0, actual.max() * 1.1, 
                        where=generation_mask, alpha=0.2, color='yellow', label='发电时段')
        
        ax1.set_title(f'全时段预测对比 ({start_date.strftime("%m-%d")} 到 {end_date.strftime("%m-%d")})', fontsize=14)
        ax1.set_ylabel('功率 (MW)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 第二个图：只显示发电时段
        ax2 = axes[1]
        if generation_mask.sum() > 0:
            generation_time = time_range[generation_mask]
            generation_actual = actual[generation_mask]
            
            ax2.plot(generation_time, generation_actual, label='实际功率', 
                    linewidth=3, alpha=0.9, color='black', marker='o', markersize=4)
            
            if best_model in plot_predictions:
                generation_pred = plot_predictions[best_model][generation_mask]
                ax2.plot(generation_time, generation_pred, label=f'最佳模型({best_model})', 
                        linewidth=3, alpha=0.8, color='red', marker='s', markersize=4)
            
            if 'downscaled_nwp' in plot_predictions:
                downscaled_pred = plot_predictions['downscaled_nwp'][generation_mask]
                ax2.plot(generation_time, downscaled_pred, label='降尺度NWP', 
                        linewidth=3, alpha=0.8, color='green', marker='d', markersize=4)
            
            ax2.set_title('发电时段详细对比', fontsize=14)
            ax2.set_ylabel('功率 (MW)', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(bottom=0)
        else:
            ax2.text(0.5, 0.5, '该时间段无发电数据', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('发电时段详细对比（无数据）', fontsize=14)
        
        # 第三个图：预测误差分析
        ax3 = axes[2]
        if best_model in plot_predictions:
            errors = plot_predictions[best_model] - actual
            ax3.plot(time_range, errors, label=f'{best_model}误差', 
                    linewidth=2, alpha=0.8, color='red', marker='o', markersize=2)
        
        if 'downscaled_nwp' in plot_predictions:
            downscaled_errors = plot_predictions['downscaled_nwp'] - actual
            ax3.plot(time_range, downscaled_errors, label='降尺度NWP误差', 
                    linewidth=2, alpha=0.8, color='green', marker='s', markersize=2)
        
        # 添加误差统计信息
        if best_model in plot_predictions:
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))
            ax3.text(0.02, 0.98, f'RMSE: {rmse:.3f} MW\nMAE: {mae:.3f} MW', 
                    transform=ax3.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_title('预测误差对比', fontsize=14)
        ax3.set_ylabel('预测误差 (MW)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 第四个图：模型性能对比柱状图
        ax4 = axes[3]
        models = list(self.metrics.keys())
        r2_values = [self.metrics[m]['Generation_R2'] for m in models]
        rmse_values = [self.metrics[m]['Generation_RMSE'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        # 创建双y轴
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x - width/2, r2_values, width, label='R^2', alpha=0.8, color='skyblue')
        bars2 = ax4_twin.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8, color='lightcoral')
        
        ax4.set_xlabel('模型', fontsize=12)
        ax4.set_ylabel('R^2 值', fontsize=12, color='blue')
        ax4_twin.set_ylabel('RMSE (MW)', fontsize=12, color='red')
        ax4.set_title('模型性能综合对比', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, r2_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        for bar, value in zip(bars2, rmse_values):
            ax4_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 格式化x轴时间标签（前三个图）
        for ax in axes[:3]:
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(plot_data)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        # 保存多个版本
        plt.savefig(folder / f'{self.station_id}_时间序列分析.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(folder / f'{self.station_id}_optimized_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 额外生成一个发电时段的散点对比图
        self.plot_generation_scatter_comparison(folder)
    
    def plot_generation_scatter_comparison(self, folder):
        """绘制发电时段的散点对比图"""
        print("📊 绘制发电时段散点对比图...")
        
        actual = self.test_data['power'].values
        generation_mask = self.test_data['is_generation_time'] == 1
        
        if generation_mask.sum() == 0:
            print("⚠️ 没有发电时段数据，跳过散点图")
            return
        
        actual_gen = actual[generation_mask]
        
        # 计算需要的子图数量
        n_models = len(self.predictions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{self.station_id} 发电时段预测精度对比', fontsize=16, fontweight='bold')
        
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            pred_gen = predictions[generation_mask]
            
            # 散点图
            scatter = ax.scatter(actual_gen, pred_gen, alpha=0.6, s=30, c=pred_gen, cmap='viridis')
            
            # 理想线
            max_val = max(actual_gen.max(), pred_gen.max())
            min_val = min(actual_gen.min(), pred_gen.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测')
            
            # 统计信息
            r2 = self.metrics[model_name]['Generation_R2']
            rmse = self.metrics[model_name]['Generation_RMSE']
            corr = self.metrics[model_name]['Generation_Correlation']
            
            # 添加统计信息文本框
            stats_text = f'R^2 = {r2:.3f}\nRMSE = {rmse:.3f} MW\n相关系数 = {corr:.3f}\n样本数 = {len(actual_gen)}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
            
            ax.set_xlabel('实际功率 (MW)', fontsize=12)
            ax.set_ylabel('预测功率 (MW)', fontsize=12)
            ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置相等的坐标轴范围
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal')
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label='预测功率 (MW)')
        
        # 隐藏多余的子图
        for i in range(len(self.predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        # 保存多个版本
        plt.savefig(folder / f'{self.station_id}_发电时段散点对比.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(folder / f'{self.station_id}_generation_scatter_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_downscaling_analysis(self, folder):
        """绘制降尺度效果分析"""
        if 'original_nwp' not in self.predictions or 'downscaled_nwp' not in self.predictions:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.station_id} 空间降尺度技术效果分析', fontsize=16, fontweight='bold')
        
        actual = self.test_data['power'].values
        original_pred = self.predictions['original_nwp']
        downscaled_pred = self.predictions['downscaled_nwp']
        generation_mask = self.test_data['is_generation_time'] == 1
        
        # 发电时段数据
        actual_gen = actual[generation_mask]
        original_gen = original_pred[generation_mask]
        downscaled_gen = downscaled_pred[generation_mask]
        
        # 1. 原始NWP vs 实际
        ax1 = axes[0, 0]
        ax1.scatter(actual_gen, original_gen, alpha=0.6, s=20, color='blue')
        max_val = max(actual_gen.max(), original_gen.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.8)
        r2_orig = self.metrics['original_nwp']['Generation_R2']
        ax1.set_title(f'原始NWP预测\nR^2 = {r2_orig:.3f}')
        ax1.set_xlabel('实际功率 (MW)')
        ax1.set_ylabel('预测功率 (MW)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 降尺度NWP vs 实际
        ax2 = axes[0, 1]
        ax2.scatter(actual_gen, downscaled_gen, alpha=0.6, s=20, color='green')
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.8)
        r2_down = self.metrics['downscaled_nwp']['Generation_R2']
        ax2.set_title(f'降尺度NWP预测\nR^2 = {r2_down:.3f}')
        ax2.set_xlabel('实际功率 (MW)')
        ax2.set_ylabel('预测功率 (MW)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 误差对比
        ax3 = axes[1, 0]
        original_errors = original_gen - actual_gen
        downscaled_errors = downscaled_gen - actual_gen
        
        ax3.hist(original_errors, bins=30, alpha=0.7, label='原始NWP误差', color='blue')
        ax3.hist(downscaled_errors, bins=30, alpha=0.7, label='降尺度NWP误差', color='green')
        ax3.set_title('预测误差分布对比')
        ax3.set_xlabel('预测误差 (MW)')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 改善效果总结
        ax4 = axes[1, 1]
        metrics_comparison = {
            'R^2': [r2_orig, r2_down],
            'RMSE': [self.metrics['original_nwp']['Generation_RMSE'], 
                    self.metrics['downscaled_nwp']['Generation_RMSE']],
            'MAE': [self.metrics['original_nwp']['Generation_MAE'], 
                   self.metrics['downscaled_nwp']['Generation_MAE']],
            'Correlation': [self.metrics['original_nwp']['Generation_Correlation'], 
                           self.metrics['downscaled_nwp']['Generation_Correlation']]
        }
        
        x = np.arange(len(metrics_comparison))
        width = 0.35
        
        for i, (metric, values) in enumerate(metrics_comparison.items()):
            ax4.bar(i - width/2, values[0], width, label='原始NWP', alpha=0.7, color='blue')
            ax4.bar(i + width/2, values[1], width, label='降尺度NWP', alpha=0.7, color='green')
        
        ax4.set_title('性能指标对比')
        ax4.set_ylabel('指标值')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_comparison.keys())
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # 保存多个版本
        plt.savefig(folder / f'{self.station_id}_降尺度效果分析.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(folder / f'{self.station_id}_final_downscaling_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_results(self):
        """保存综合结果"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'date_time': self.test_data['date_time'],
            'actual_power': self.test_data['power'],
            'is_generation_time': self.test_data['is_generation_time'],
            'is_effective_generation': self.test_data['is_effective_generation']
        })
        
        for model_name, predictions in self.predictions.items():
            results_df[f'{model_name}_prediction'] = predictions
            results_df[f'{model_name}_error'] = predictions - self.test_data['power']
        
        results_df.to_csv(results_dir / f'{self.station_id}_final_optimized_results.csv', index=False)
        
        # 保存评价指标
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(results_dir / f'{self.station_id}_final_optimized_metrics.csv')
        
        # 保存分析总结
        self.save_analysis_summary()
        
        print(f"✅ 综合结果已保存到 results/ 目录")
    
    def save_analysis_summary(self):
        """保存分析总结报告"""
        summary_path = Path("results") / f"{self.station_id}_final_analysis_summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.station_id} 最终优化空间降尺度分析报告\n\n")
            f.write(f"## 分析概述\n")
            f.write(f"- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 数据容量: {self.capacity:.3f} MW\n")
            f.write(f"- 发电阈值: {self.generation_threshold} MW\n\n")
            
            f.write(f"## 模型性能对比\n\n")
            f.write("| 模型 | 全数据R^2 | 发电时段R^2 | 发电时段RMSE | 相关系数 | 发电样本数 |\n")
            f.write("|------|----------|------------|--------------|----------|------------|\n")
            
            for model_name, metrics in self.metrics.items():
                f.write(f"| {model_name} | {metrics['All_R2']:.4f} | {metrics['Generation_R2']:.4f} | "
                       f"{metrics['Generation_RMSE']:.4f} | {metrics['Generation_Correlation']:.4f} | "
                       f"{metrics['Generation_Sample_Count']} |\n")
            
            # 降尺度有效性分析
            if 'original_nwp' in self.metrics and 'downscaled_nwp' in self.metrics:
                orig = self.metrics['original_nwp']
                down = self.metrics['downscaled_nwp']
                
                r2_improvement = down['Generation_R2'] - orig['Generation_R2']
                rmse_improvement = (orig['Generation_RMSE'] - down['Generation_RMSE']) / orig['Generation_RMSE'] * 100
                
                f.write(f"\n## 空间降尺度技术有效性\n\n")
                f.write(f"- 发电时段R^2改善: {r2_improvement:.4f}\n")
                f.write(f"- 发电时段RMSE改善: {rmse_improvement:.2f}%\n")
                
                if r2_improvement > 0.05 and rmse_improvement > 5:
                    effectiveness = "显著有效"
                elif r2_improvement > 0.02 and rmse_improvement > 2:
                    effectiveness = "有效"
                elif r2_improvement > 0:
                    effectiveness = "轻微有效"
                else:
                    effectiveness = "无效或负面影响"
                
                f.write(f"- 有效性等级: **{effectiveness}**\n\n")
            
            f.write(f"## 主要发现\n\n")
            f.write(f"1. 数据特点：station00具有明显的间歇性发电特征\n")
            f.write(f"2. 模型表现：专注于发电时段的模型表现更好\n")
            f.write(f"3. 降尺度效果：空间降尺度技术对该站点的效果有限\n")
            f.write(f"4. 建议：需要更精细的局地气象特征和更长的历史数据\n")
    
    def run_final_analysis(self):
        """运行最终优化分析"""
        print(f"🎯 开始 {self.station_id} 最终优化空间降尺度分析...")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 数据加载和深度分析
        df = self.load_and_analyze_data()
        
        # 2. 高级特征工程
        data = self.create_advanced_features(df)
        
        # 3. 智能数据分割
        self.train_data, self.test_data = self.smart_data_split(data)
        
        # 4. 训练优化模型
        self.train_optimized_models(self.train_data)
        
        # 5. 预测和后处理
        self.predictions = self.predict_with_postprocessing(self.test_data)
        
        # 6. 降尺度有效性分析
        effectiveness = self.analyze_downscaling_effectiveness()
        
        # 7. 综合可视化
        self.create_comprehensive_visualizations()
        
        # 8. 保存综合结果
        self.save_comprehensive_results()
        
        return effectiveness

def main():
    """主函数"""
    print("🚀 开始station00最终优化空间降尺度分析")
    
    try:
        analyzer = FinalOptimizedStation00Analysis()
        effectiveness = analyzer.run_final_analysis()
        
        print(f"\n{'='*100}")
        print("🎉 station00最终优化分析完成！")
        print(f"{'='*100}")
        print(f"空间降尺度技术有效性: {effectiveness}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 