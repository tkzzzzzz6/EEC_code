# 融入NWP信息的光伏发电功率预测模型 - 问题3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# 简化的中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')
plt.rcParams['axes.unicode_minus'] = False

sns.set_palette("husl")
plt.ioff()

class NWPPowerPrediction:
    """融入NWP信息的光伏发电功率预测模型"""
    
    def __init__(self, station_id: str):
        self.station_id = station_id
        self.models = {}  # 存储不同模型
        self.feature_names = []
        self.capacity = None
        self.train_data = None
        self.test_data = None
        self.predictions = {}  # 存储不同模型的预测结果
        self.metrics = {}  # 存储不同模型的评价指标
        
    def ensure_chinese_font(self):
        """确保中文字体设置正确应用"""
        mpl.rc('font', family='simhei')
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print(f"📊 加载 {self.station_id} 数据...")
        
        # 加载数据
        df = pd.read_csv(f'data/{self.station_id}.csv')
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        
        # 估算开机容量
        self.capacity = df['power'].max() * 1.1
        
        print(f"  数据时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
        print(f"  数据点数: {len(df):,}")
        print(f"  平均功率: {df['power'].mean():.3f} MW")
        print(f"  最大功率: {df['power'].max():.3f} MW")
        print(f"  估算开机容量: {self.capacity:.3f} MW")
        
        # 检查NWP和LMD数据的可用性
        nwp_cols = [col for col in df.columns if 'nwp_' in col]
        lmd_cols = [col for col in df.columns if 'lmd_' in col]
        
        print(f"  NWP字段数量: {len(nwp_cols)}")
        print(f"  LMD字段数量: {len(lmd_cols)}")
        
        # 检查数据质量
        print(f"  NWP数据缺失率: {df[nwp_cols].isnull().sum().sum() / (len(df) * len(nwp_cols)) * 100:.2f}%")
        print(f"  LMD数据缺失率: {df[lmd_cols].isnull().sum().sum() / (len(df) * len(lmd_cols)) * 100:.2f}%")
        
        return df
    
    def create_features(self, df):
        """创建特征工程 - 包含NWP和LMD特征"""
        print("🔧 创建特征工程（包含NWP信息）...")
        
        data = df.copy()
        
        # 1. 时间特征
        data['hour'] = data['date_time'].dt.hour
        data['minute'] = data['date_time'].dt.minute
        data['day_of_week'] = data['date_time'].dt.dayofweek
        data['day_of_year'] = data['date_time'].dt.dayofyear
        data['month'] = data['date_time'].dt.month
        data['time_slot'] = data['hour'] * 4 + data['minute'] // 15
        
        # 2. 周期性特征
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['time_slot_sin'] = np.sin(2 * np.pi * data['time_slot'] / 96)
        data['time_slot_cos'] = np.cos(2 * np.pi * data['time_slot'] / 96)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # 3. NWP特征工程
        # 3.1 NWP原始特征
        nwp_features = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                       'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
        
        # 3.2 NWP衍生特征
        # 辐射相关特征
        data['nwp_diffuse_irrad'] = data['nwp_globalirrad'] - data['nwp_directirrad']
        data['nwp_clearness_index'] = data['nwp_globalirrad'] / (data['nwp_globalirrad'].max() + 1e-6)
        data['nwp_direct_ratio'] = data['nwp_directirrad'] / (data['nwp_globalirrad'] + 1e-6)
        
        # 温度相关特征
        data['nwp_temp_celsius'] = data['nwp_temperature'] - 273.15  # 转换为摄氏度
        data['nwp_temp_optimal'] = np.abs(data['nwp_temp_celsius'] - 25)  # 与最优温度的差异
        
        # 风速风向特征
        data['nwp_wind_u'] = data['nwp_windspeed'] * np.cos(np.radians(data['nwp_winddirection']))
        data['nwp_wind_v'] = data['nwp_windspeed'] * np.sin(np.radians(data['nwp_winddirection']))
        
        # 3.3 NWP滞后特征
        nwp_lag_periods = [1, 2, 4, 8, 12, 24]
        for feature in ['nwp_globalirrad', 'nwp_temperature', 'nwp_humidity']:
            for lag in nwp_lag_periods:
                data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
        
        # 3.4 NWP滚动统计特征
        nwp_windows = [4, 8, 12, 24]
        for feature in ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature']:
            for window in nwp_windows:
                data[f'{feature}_rolling_mean_{window}'] = data[feature].rolling(window=window).mean()
                data[f'{feature}_rolling_std_{window}'] = data[feature].rolling(window=window).std()
        
        # 4. LMD特征工程
        # 4.1 LMD原始特征
        lmd_features = ['lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 
                       'lmd_pressure', 'lmd_winddirection', 'lmd_windspeed']
        
        # 4.2 LMD衍生特征
        data['lmd_direct_irrad'] = data['lmd_totalirrad'] - data['lmd_diffuseirrad']
        data['lmd_clearness_index'] = data['lmd_totalirrad'] / (data['lmd_totalirrad'].max() + 1e-6)
        data['lmd_diffuse_ratio'] = data['lmd_diffuseirrad'] / (data['lmd_totalirrad'] + 1e-6)
        
        # LMD风速风向特征
        data['lmd_wind_u'] = data['lmd_windspeed'] * np.cos(np.radians(data['lmd_winddirection']))
        data['lmd_wind_v'] = data['lmd_windspeed'] * np.sin(np.radians(data['lmd_winddirection']))
        
        # 5. NWP与LMD的对比特征
        data['irrad_nwp_lmd_diff'] = data['nwp_globalirrad'] - data['lmd_totalirrad']
        data['temp_nwp_lmd_diff'] = data['nwp_temperature'] - data['lmd_temperature']
        data['pressure_nwp_lmd_diff'] = data['nwp_pressure'] - data['lmd_pressure']
        data['windspeed_nwp_lmd_diff'] = data['nwp_windspeed'] - data['lmd_windspeed']
        
        # NWP与LMD的相关性特征
        data['irrad_nwp_lmd_ratio'] = data['nwp_globalirrad'] / (data['lmd_totalirrad'] + 1e-6)
        data['temp_nwp_lmd_ratio'] = data['nwp_temperature'] / (data['lmd_temperature'] + 273.15)
        
        # 6. 功率相关特征
        # 功率滞后特征
        power_lag_periods = [1, 2, 3, 4, 8, 12, 24, 48, 96]
        for lag in power_lag_periods:
            data[f'power_lag_{lag}'] = data['power'].shift(lag)
        
        # 功率滚动统计特征
        power_windows = [4, 8, 12, 24, 48]
        for window in power_windows:
            data[f'power_rolling_mean_{window}'] = data['power'].rolling(window=window).mean()
            data[f'power_rolling_std_{window}'] = data['power'].rolling(window=window).std()
            data[f'power_rolling_max_{window}'] = data['power'].rolling(window=window).max()
        
        # 7. 白天判断和太阳角度特征
        data['is_daytime'] = ((data['hour'] >= 6) & (data['hour'] <= 18)).astype(int)
        
        # 太阳高度角近似计算（简化版）
        day_of_year = data['day_of_year']
        hour_angle = (data['hour'] - 12) * 15  # 时角
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        latitude = 38.0  # 假设纬度（根据metadata调整）
        
        data['solar_elevation'] = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) * 
            np.cos(np.radians(hour_angle))
        )
        data['solar_elevation'] = np.degrees(data['solar_elevation'])
        data['solar_elevation'] = np.maximum(data['solar_elevation'], 0)  # 负值设为0
        
        # 8. 天气状况分类特征
        # 基于NWP数据的天气分类
        data['weather_clear'] = ((data['nwp_globalirrad'] > 600) & 
                                (data['nwp_humidity'] < 70)).astype(int)
        data['weather_cloudy'] = ((data['nwp_globalirrad'] > 200) & 
                                 (data['nwp_globalirrad'] <= 600)).astype(int)
        data['weather_overcast'] = (data['nwp_globalirrad'] <= 200).astype(int)
        data['weather_high_humidity'] = (data['nwp_humidity'] > 80).astype(int)
        
        # 9. 填充缺失值和处理异常值
        data = data.fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
        
        # 处理极值
        for col in data.columns:
            if col not in ['date_time'] and data[col].dtype in ['float64', 'int64']:
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                data[col] = data[col].clip(lower=q01, upper=q99)
        
        print(f"  总特征数量: {len(data.columns) - 2}")  # 减去date_time和power
        
        # 分析特征重要性类别
        nwp_feature_count = len([col for col in data.columns if 'nwp_' in col])
        lmd_feature_count = len([col for col in data.columns if 'lmd_' in col])
        time_feature_count = len([col for col in data.columns if any(x in col for x in ['hour', 'day', 'month', 'time_slot', 'sin', 'cos'])])
        power_feature_count = len([col for col in data.columns if 'power_' in col])
        
        print(f"  NWP相关特征: {nwp_feature_count}")
        print(f"  LMD相关特征: {lmd_feature_count}")
        print(f"  时间相关特征: {time_feature_count}")
        print(f"  功率历史特征: {power_feature_count}")
        
        return data
    
    def split_data(self, data, test_days=7):
        """分割训练和测试数据"""
        print(f"📊 分割数据 - 最后{test_days}天作为测试集...")
        
        test_start_idx = len(data) - test_days * 96
        
        train_data = data[:test_start_idx].copy()
        test_data = data[test_start_idx:].copy()
        
        print(f"  训练集: {len(train_data):,} 条记录")
        print(f"  测试集: {len(test_data):,} 条记录")
        
        return train_data, test_data
    
    def train_models(self, train_data):
        """训练多个模型进行对比"""
        print("🚀 训练多个预测模型...")
        
        # 只使用白天时段的数据进行训练
        daytime_mask = train_data['is_daytime'] == 1
        train_subset = train_data[daytime_mask].copy()
        train_subset = train_subset.dropna()
        
        if len(train_subset) == 0:
            raise ValueError("没有足够的训练数据")
        
        # 准备特征和目标变量
        feature_cols = [col for col in train_subset.columns 
                       if col not in ['date_time', 'power']]
        X_train = train_subset[feature_cols]
        y_train = train_subset['power']
        
        self.feature_names = feature_cols
        
        # 1. 基础模型（不使用NWP）
        print("  训练基础模型（不使用NWP）...")
        basic_features = [col for col in feature_cols if not col.startswith('nwp_')]
        X_train_basic = X_train[basic_features]
        
        self.models['basic'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=8,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        self.models['basic'].fit(X_train_basic, y_train)
        
        # 2. NWP增强模型（使用所有特征）
        print("  训练NWP增强模型...")
        self.models['nwp_enhanced'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=10,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        self.models['nwp_enhanced'].fit(X_train, y_train)
        
        # 3. 只使用NWP的模型
        print("  训练纯NWP模型...")
        nwp_features = [col for col in feature_cols if 'nwp_' in col or 
                       col in ['hour', 'minute', 'day_of_week', 'month', 'is_daytime',
                              'hour_sin', 'hour_cos', 'solar_elevation']]
        X_train_nwp = X_train[nwp_features]
        
        self.models['nwp_only'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=8,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        self.models['nwp_only'].fit(X_train_nwp, y_train)
        
        # 4. 随机森林模型（用于对比）
        print("  训练随机森林模型...")
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # 打印训练集性能
        for model_name, model in self.models.items():
            if model_name == 'basic':
                train_pred = model.predict(X_train_basic)
            elif model_name == 'nwp_only':
                train_pred = model.predict(X_train_nwp)
            else:
                train_pred = model.predict(X_train)
            
            train_r2 = r2_score(y_train, train_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            print(f"    {model_name} - R^2: {train_r2:.4f}, MAE: {train_mae:.4f}")
    
    def predict_test_period(self, test_data):
        """使用不同模型预测测试期间的功率"""
        print("🔮 使用不同模型预测测试期间功率...")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            model_predictions = []
            
            for idx, row in test_data.iterrows():
                # 准备特征
                if model_name == 'basic':
                    feature_cols = [col for col in self.feature_names if not col.startswith('nwp_')]
                elif model_name == 'nwp_only':
                    feature_cols = [col for col in self.feature_names if 'nwp_' in col or 
                                   col in ['hour', 'minute', 'day_of_week', 'month', 'is_daytime',
                                          'hour_sin', 'hour_cos', 'solar_elevation']]
                else:
                    feature_cols = self.feature_names
                
                features = row[feature_cols].values.reshape(1, -1)
                
                # 预测
                pred = model.predict(features)[0]
                pred = max(0, pred)
                
                # 夜间时段设为0
                if row['is_daytime'] == 0:
                    pred = 0
                else:
                    # 添加适当的随机扰动
                    noise_factor = np.random.normal(0, 0.02)
                    pred = pred * (1 + noise_factor)
                    pred = max(0, min(pred, self.capacity))
                
                model_predictions.append(pred)
            
            predictions[model_name] = np.array(model_predictions)
            print(f"  {model_name} 预测完成")
        
        return predictions
    
    def calculate_evaluation_metrics(self, actual, predicted):
        """计算评价指标"""
        daytime_mask = (actual > 0) | (predicted > 0)
        
        if np.sum(daytime_mask) == 0:
            return {}
        
        actual_day = actual[daytime_mask]
        predicted_day = predicted[daytime_mask]
        
        # 归一化误差
        normalized_actual = actual_day / self.capacity
        normalized_predicted = predicted_day / self.capacity
        normalized_error = normalized_predicted - normalized_actual
        
        rmse = np.sqrt(np.mean(normalized_error ** 2))
        mae = np.mean(np.abs(normalized_error))
        me = np.mean(normalized_error)
        
        if len(actual_day) > 1:
            correlation = np.corrcoef(actual_day, predicted_day)[0, 1]
        else:
            correlation = 0
        
        accuracy = (1 - rmse) * 100
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
    
    def analyze_nwp_effectiveness(self):
        """分析NWP信息的有效性"""
        print("📈 分析NWP信息的有效性...")
        
        actual = self.test_data['power'].values
        
        # 计算各模型的评价指标
        for model_name, predictions in self.predictions.items():
            metrics = self.calculate_evaluation_metrics(actual, predictions)
            self.metrics[model_name] = metrics
        
        # 对比分析
        print("\n" + "="*60)
        print("📊 模型性能对比分析")
        print("="*60)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()} 模型:")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  相关系数: {metrics['Correlation']:.4f}")
            print(f"  准确率: {metrics['Accuracy']:.2f}%")
            print(f"  合格率: {metrics['Qualification_Rate']:.2f}%")
        
        # NWP有效性分析
        print(f"\n{'='*60}")
        print("🎯 NWP信息有效性分析")
        print("="*60)
        
        basic_rmse = self.metrics['basic']['RMSE']
        nwp_enhanced_rmse = self.metrics['nwp_enhanced']['RMSE']
        nwp_only_rmse = self.metrics['nwp_only']['RMSE']
        
        rmse_improvement = (basic_rmse - nwp_enhanced_rmse) / basic_rmse * 100
        
        print(f"基础模型 vs NWP增强模型:")
        print(f"  RMSE改善: {rmse_improvement:.2f}%")
        
        basic_acc = self.metrics['basic']['Accuracy']
        nwp_enhanced_acc = self.metrics['nwp_enhanced']['Accuracy']
        acc_improvement = nwp_enhanced_acc - basic_acc
        
        print(f"  准确率提升: {acc_improvement:.2f}个百分点")
        
        # 判断NWP是否有效
        if rmse_improvement > 5 and acc_improvement > 2:
            effectiveness = "显著有效"
        elif rmse_improvement > 2 and acc_improvement > 1:
            effectiveness = "有效"
        elif rmse_improvement > 0:
            effectiveness = "轻微有效"
        else:
            effectiveness = "无效或负面影响"
        
        print(f"\n🎯 NWP信息有效性评价: {effectiveness}")
        
        return effectiveness
    
    def identify_improvement_scenarios(self):
        """识别NWP信息提高预测精度的场景"""
        print("\n🔍 识别NWP信息提高预测精度的场景...")
        
        # 准备数据
        test_data_copy = self.test_data.copy()
        test_data_copy['actual_power'] = test_data_copy['power']
        test_data_copy['basic_pred'] = self.predictions['basic']
        test_data_copy['nwp_enhanced_pred'] = self.predictions['nwp_enhanced']
        
        # 计算误差
        test_data_copy['basic_error'] = np.abs(test_data_copy['basic_pred'] - test_data_copy['actual_power'])
        test_data_copy['nwp_enhanced_error'] = np.abs(test_data_copy['nwp_enhanced_pred'] - test_data_copy['actual_power'])
        test_data_copy['error_improvement'] = test_data_copy['basic_error'] - test_data_copy['nwp_enhanced_error']
        
        # 只分析白天时段
        daytime_data = test_data_copy[test_data_copy['is_daytime'] == 1].copy()
        
        if len(daytime_data) == 0:
            print("❌ 没有白天时段数据进行场景分析")
            return {}
        
        # 场景分析
        scenarios = {}
        
        # 1. 天气条件场景
        print("\n1️⃣ 天气条件场景分析:")
        
        # 晴天场景
        clear_mask = daytime_data['weather_clear'] == 1
        if clear_mask.sum() > 0:
            clear_improvement = daytime_data[clear_mask]['error_improvement'].mean()
            scenarios['clear_weather'] = {
                'improvement': clear_improvement,
                'sample_count': clear_mask.sum(),
                'description': '晴天场景'
            }
            print(f"  晴天: 平均误差改善 {clear_improvement:.4f} MW ({clear_mask.sum()}个样本)")
        
        # 多云场景
        cloudy_mask = daytime_data['weather_cloudy'] == 1
        if cloudy_mask.sum() > 0:
            cloudy_improvement = daytime_data[cloudy_mask]['error_improvement'].mean()
            scenarios['cloudy_weather'] = {
                'improvement': cloudy_improvement,
                'sample_count': cloudy_mask.sum(),
                'description': '多云场景'
            }
            print(f"  多云: 平均误差改善 {cloudy_improvement:.4f} MW ({cloudy_mask.sum()}个样本)")
        
        # 阴天场景
        overcast_mask = daytime_data['weather_overcast'] == 1
        if overcast_mask.sum() > 0:
            overcast_improvement = daytime_data[overcast_mask]['error_improvement'].mean()
            scenarios['overcast_weather'] = {
                'improvement': overcast_improvement,
                'sample_count': overcast_mask.sum(),
                'description': '阴天场景'
            }
            print(f"  阴天: 平均误差改善 {overcast_improvement:.4f} MW ({overcast_mask.sum()}个样本)")
        
        # 2. 辐射强度场景
        print("\n2️⃣ 辐射强度场景分析:")
        
        # 高辐射场景
        high_irrad_mask = daytime_data['nwp_globalirrad'] > daytime_data['nwp_globalirrad'].quantile(0.75)
        if high_irrad_mask.sum() > 0:
            high_irrad_improvement = daytime_data[high_irrad_mask]['error_improvement'].mean()
            scenarios['high_irradiance'] = {
                'improvement': high_irrad_improvement,
                'sample_count': high_irrad_mask.sum(),
                'description': '高辐射场景'
            }
            print(f"  高辐射: 平均误差改善 {high_irrad_improvement:.4f} MW ({high_irrad_mask.sum()}个样本)")
        
        # 中等辐射场景
        med_irrad_mask = ((daytime_data['nwp_globalirrad'] > daytime_data['nwp_globalirrad'].quantile(0.25)) & 
                         (daytime_data['nwp_globalirrad'] <= daytime_data['nwp_globalirrad'].quantile(0.75)))
        if med_irrad_mask.sum() > 0:
            med_irrad_improvement = daytime_data[med_irrad_mask]['error_improvement'].mean()
            scenarios['medium_irradiance'] = {
                'improvement': med_irrad_improvement,
                'sample_count': med_irrad_mask.sum(),
                'description': '中等辐射场景'
            }
            print(f"  中等辐射: 平均误差改善 {med_irrad_improvement:.4f} MW ({med_irrad_mask.sum()}个样本)")
        
        # 低辐射场景
        low_irrad_mask = daytime_data['nwp_globalirrad'] <= daytime_data['nwp_globalirrad'].quantile(0.25)
        if low_irrad_mask.sum() > 0:
            low_irrad_improvement = daytime_data[low_irrad_mask]['error_improvement'].mean()
            scenarios['low_irradiance'] = {
                'improvement': low_irrad_improvement,
                'sample_count': low_irrad_mask.sum(),
                'description': '低辐射场景'
            }
            print(f"  低辐射: 平均误差改善 {low_irrad_improvement:.4f} MW ({low_irrad_mask.sum()}个样本)")
        
        # 3. 时间场景
        print("\n3️⃣ 时间场景分析:")
        
        # 上午场景
        morning_mask = (daytime_data['hour'] >= 6) & (daytime_data['hour'] < 12)
        if morning_mask.sum() > 0:
            morning_improvement = daytime_data[morning_mask]['error_improvement'].mean()
            scenarios['morning'] = {
                'improvement': morning_improvement,
                'sample_count': morning_mask.sum(),
                'description': '上午时段'
            }
            print(f"  上午(6-12h): 平均误差改善 {morning_improvement:.4f} MW ({morning_mask.sum()}个样本)")
        
        # 下午场景
        afternoon_mask = (daytime_data['hour'] >= 12) & (daytime_data['hour'] <= 18)
        if afternoon_mask.sum() > 0:
            afternoon_improvement = daytime_data[afternoon_mask]['error_improvement'].mean()
            scenarios['afternoon'] = {
                'improvement': afternoon_improvement,
                'sample_count': afternoon_mask.sum(),
                'description': '下午时段'
            }
            print(f"  下午(12-18h): 平均误差改善 {afternoon_improvement:.4f} MW ({afternoon_mask.sum()}个样本)")
        
        # 4. 温度场景
        print("\n4️⃣ 温度场景分析:")
        
        # 高温场景
        high_temp_mask = daytime_data['nwp_temp_celsius'] > daytime_data['nwp_temp_celsius'].quantile(0.75)
        if high_temp_mask.sum() > 0:
            high_temp_improvement = daytime_data[high_temp_mask]['error_improvement'].mean()
            scenarios['high_temperature'] = {
                'improvement': high_temp_improvement,
                'sample_count': high_temp_mask.sum(),
                'description': '高温场景'
            }
            print(f"  高温: 平均误差改善 {high_temp_improvement:.4f} MW ({high_temp_mask.sum()}个样本)")
        
        # 适宜温度场景
        opt_temp_mask = daytime_data['nwp_temp_optimal'] < 5  # 与25°C相差小于5°C
        if opt_temp_mask.sum() > 0:
            opt_temp_improvement = daytime_data[opt_temp_mask]['error_improvement'].mean()
            scenarios['optimal_temperature'] = {
                'improvement': opt_temp_improvement,
                'sample_count': opt_temp_mask.sum(),
                'description': '适宜温度场景'
            }
            print(f"  适宜温度: 平均误差改善 {opt_temp_improvement:.4f} MW ({opt_temp_mask.sum()}个样本)")
        
        # 找出最有效的场景
        print(f"\n{'='*60}")
        print("🏆 NWP信息最有效的场景排名")
        print("="*60)
        
        sorted_scenarios = sorted(scenarios.items(), 
                                key=lambda x: x[1]['improvement'], reverse=True)
        
        for i, (scenario_name, scenario_data) in enumerate(sorted_scenarios[:5], 1):
            print(f"{i}. {scenario_data['description']}: "
                  f"误差改善 {scenario_data['improvement']:.4f} MW "
                  f"({scenario_data['sample_count']}个样本)")
        
        return scenarios
    
    def create_comprehensive_visualizations(self):
        """创建综合可视化分析"""
        print("📊 生成综合可视化分析...")
        
        self.ensure_chinese_font()
        
        fig_dir = Path("results/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 模型性能对比图
        self.plot_model_comparison()
        
        # 2. NWP特征重要性分析
        self.plot_nwp_feature_importance()
        
        # 3. 场景分析可视化
        scenarios = self.identify_improvement_scenarios()
        self.plot_scenario_analysis(scenarios)
        
        # 4. 时间序列对比
        self.plot_time_series_comparison()
        
        print("✅ 可视化分析完成")
    
    def plot_model_comparison(self):
        """绘制模型性能对比图"""
        self.ensure_chinese_font()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.station_id} 不同模型性能对比', fontsize=16, fontweight='bold')
        
        models = list(self.metrics.keys())
        
        # RMSE对比
        ax1 = axes[0, 0]
        rmse_values = [self.metrics[m]['RMSE'] for m in models]
        bars1 = ax1.bar(models, rmse_values, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
        ax1.set_title('RMSE对比 (越小越好)')
        ax1.set_ylabel('RMSE')
        ax1.grid(True, alpha=0.3)
        for bar, value in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 准确率对比
        ax2 = axes[0, 1]
        acc_values = [self.metrics[m]['Accuracy'] for m in models]
        bars2 = ax2.bar(models, acc_values, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
        ax2.set_title('准确率对比 (越大越好)')
        ax2.set_ylabel('准确率 (%)')
        ax2.grid(True, alpha=0.3)
        for bar, value in zip(bars2, acc_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 相关系数对比
        ax3 = axes[1, 0]
        corr_values = [self.metrics[m]['Correlation'] for m in models]
        bars3 = ax3.bar(models, corr_values, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
        ax3.set_title('相关系数对比 (越大越好)')
        ax3.set_ylabel('相关系数')
        ax3.grid(True, alpha=0.3)
        for bar, value in zip(bars3, corr_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 合格率对比
        ax4 = axes[1, 1]
        qr_values = [self.metrics[m]['Qualification_Rate'] for m in models]
        bars4 = ax4.bar(models, qr_values, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
        ax4.set_title('合格率对比 (越大越好)')
        ax4.set_ylabel('合格率 (%)')
        ax4.grid(True, alpha=0.3)
        for bar, value in zip(bars4, qr_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'results/figures/{self.station_id}_model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_nwp_feature_importance(self):
        """绘制NWP特征重要性"""
        self.ensure_chinese_font()
        
        if hasattr(self.models['nwp_enhanced'], 'feature_importances_'):
            importances = self.models['nwp_enhanced'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # 分类特征
            nwp_features = feature_importance_df[feature_importance_df['feature'].str.contains('nwp_')]
            lmd_features = feature_importance_df[feature_importance_df['feature'].str.contains('lmd_')]
            other_features = feature_importance_df[~feature_importance_df['feature'].str.contains('nwp_|lmd_')]
            
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            fig.suptitle(f'{self.station_id} 特征重要性分析', fontsize=16, fontweight='bold')
            
            # NWP特征重要性
            ax1 = axes[0]
            top_nwp = nwp_features.head(15)
            sns.barplot(data=top_nwp, y='feature', x='importance', ax=ax1)
            ax1.set_title('NWP特征重要性 (Top 15)')
            ax1.set_xlabel('重要性得分')
            
            # LMD特征重要性
            ax2 = axes[1]
            top_lmd = lmd_features.head(10)
            sns.barplot(data=top_lmd, y='feature', x='importance', ax=ax2)
            ax2.set_title('LMD特征重要性 (Top 10)')
            ax2.set_xlabel('重要性得分')
            
            # 其他特征重要性
            ax3 = axes[2]
            top_other = other_features.head(10)
            sns.barplot(data=top_other, y='feature', x='importance', ax=ax3)
            ax3.set_title('其他特征重要性 (Top 10)')
            ax3.set_xlabel('重要性得分')
            
            plt.tight_layout()
            plt.savefig(f'results/figures/{self.station_id}_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_scenario_analysis(self, scenarios):
        """绘制场景分析图"""
        self.ensure_chinese_font()
        
        if not scenarios:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scenario_names = [scenarios[k]['description'] for k in scenarios.keys()]
        improvements = [scenarios[k]['improvement'] for k in scenarios.keys()]
        sample_counts = [scenarios[k]['sample_count'] for k in scenarios.keys()]
        
        # 创建气泡图
        colors = plt.cm.viridis(np.linspace(0, 1, len(scenario_names)))
        scatter = ax.scatter(range(len(scenario_names)), improvements, 
                           s=[c*10 for c in sample_counts], c=colors, alpha=0.7)
        
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.set_ylabel('误差改善 (MW)')
        ax.set_title(f'{self.station_id} NWP信息在不同场景下的有效性')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 添加数值标签
        for i, (improvement, count) in enumerate(zip(improvements, sample_counts)):
            ax.text(i, improvement + 0.01, f'{improvement:.3f}\n({count}样本)', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'results/figures/{self.station_id}_scenario_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series_comparison(self):
        """绘制时间序列对比"""
        self.ensure_chinese_font()
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(f'{self.station_id} 预测结果时间序列对比', fontsize=16, fontweight='bold')
        
        time_range = self.test_data['date_time']
        actual = self.test_data['power']
        
        # 上图：所有模型对比
        ax1 = axes[0]
        ax1.plot(time_range, actual, label='实际功率', linewidth=2, alpha=0.8, color='black')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            ax1.plot(time_range, predictions, label=f'{model_name}预测', 
                    linewidth=1.5, alpha=0.7, color=colors[i])
        
        ax1.set_title('所有模型预测对比')
        ax1.set_ylabel('功率 (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：基础模型vs NWP增强模型
        ax2 = axes[1]
        ax2.plot(time_range, actual, label='实际功率', linewidth=2, alpha=0.8, color='black')
        ax2.plot(time_range, self.predictions['basic'], label='基础模型', 
                linewidth=1.5, alpha=0.7, color='red')
        ax2.plot(time_range, self.predictions['nwp_enhanced'], label='NWP增强模型', 
                linewidth=1.5, alpha=0.7, color='blue')
        
        ax2.set_title('基础模型 vs NWP增强模型')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('功率 (MW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴
        for ax in axes:
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'results/figures/{self.station_id}_time_series_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """保存所有结果"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'date_time': self.test_data['date_time'],
            'actual_power': self.test_data['power']
        })
        
        for model_name, predictions in self.predictions.items():
            results_df[f'{model_name}_prediction'] = predictions
            results_df[f'{model_name}_error'] = predictions - self.test_data['power']
        
        results_df.to_csv(results_dir / f'{self.station_id}_nwp_prediction_results.csv', index=False)
        
        # 保存模型
        for model_name, model in self.models.items():
            joblib.dump(model, results_dir / f'{self.station_id}_{model_name}_model.pkl')
        
        # 保存评价指标
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(results_dir / f'{self.station_id}_nwp_metrics.csv')
        
        print(f"✅ 所有结果已保存到 results/ 目录")
    
    def run_complete_analysis(self):
        """运行完整的NWP分析流程"""
        print(f"🎯 开始 {self.station_id} NWP信息融入分析...")
        
        # 1. 加载数据
        df = self.load_and_preprocess_data()
        
        # 2. 特征工程
        data = self.create_features(df)
        
        # 3. 分割数据
        self.train_data, self.test_data = self.split_data(data, test_days=7)
        
        # 4. 训练多个模型
        self.train_models(self.train_data)
        
        # 5. 预测
        self.predictions = self.predict_test_period(self.test_data)
        
        # 6. 分析NWP有效性
        effectiveness = self.analyze_nwp_effectiveness()
        
        # 7. 场景分析
        scenarios = self.identify_improvement_scenarios()
        
        # 8. 可视化
        self.create_comprehensive_visualizations()
        
        # 9. 保存结果
        self.save_results()
        
        return effectiveness, scenarios

def run_nwp_analysis():
    """运行NWP信息融入分析"""
    stations = ['station00', 'station04', 'station05', 'station09']
    all_results = {}
    
    print("🚀 开始NWP信息融入的光伏发电功率预测分析...")
    print("="*80)
    
    for station in stations:
        try:
            print(f"\n{'='*30} {station} {'='*30}")
            predictor = NWPPowerPrediction(station)
            effectiveness, scenarios = predictor.run_complete_analysis()
            
            all_results[station] = {
                'effectiveness': effectiveness,
                'scenarios': scenarios,
                'metrics': predictor.metrics
            }
            
        except Exception as e:
            print(f"❌ {station} 分析失败: {str(e)}")
            continue
    
    # 生成综合报告
    if all_results:
        create_nwp_summary_report(all_results)
    
    print(f"\n🎉 NWP信息融入分析完成！")
    return all_results

def create_nwp_summary_report(all_results):
    """创建NWP分析综合报告"""
    print(f"\n{'='*80}")
    print("📊 NWP信息融入效果综合报告")
    print("="*80)
    
    # 统计有效性
    effectiveness_count = {}
    for station, results in all_results.items():
        effectiveness = results['effectiveness']
        effectiveness_count[effectiveness] = effectiveness_count.get(effectiveness, 0) + 1
        print(f"{station}: {effectiveness}")
    
    print(f"\n📈 有效性统计:")
    for effectiveness, count in effectiveness_count.items():
        print(f"  {effectiveness}: {count}个站点")
    
    # 保存综合报告
    report = f"""
# NWP信息融入光伏发电功率预测综合分析报告

## 分析概述
本报告分析了NWP（数值天气预报）信息对光伏发电功率预测精度的影响。

## 站点分析结果

"""
    
    for station, results in all_results.items():
        report += f"""
### {station}
- **NWP有效性**: {results['effectiveness']}
- **模型性能对比**:
"""
        
        for model_name, metrics in results['metrics'].items():
            report += f"  - {model_name}: RMSE={metrics['RMSE']:.6f}, 准确率={metrics['Accuracy']:.2f}%\n"
        
        if results['scenarios']:
            report += f"- **最有效场景**: {list(results['scenarios'].keys())[:3]}\n"
    
    report += f"""
## 总体结论

### NWP信息有效性统计
"""
    
    for effectiveness, count in effectiveness_count.items():
        report += f"- {effectiveness}: {count}个站点\n"
    
    report += f"""
### 建议
1. 对于NWP信息显著有效的站点，建议在实际应用中采用NWP增强模型
2. 对于NWP信息有效的站点，可在特定场景下使用NWP信息
3. 对于NWP信息无效的站点，建议优化NWP数据质量或特征工程方法

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open('results/nwp_comprehensive_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 综合报告已保存: results/nwp_comprehensive_report.md")

if __name__ == "__main__":
    # 运行NWP信息融入分析
    results = run_nwp_analysis() 