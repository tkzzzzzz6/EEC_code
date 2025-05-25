# 简化版光伏功率预测演示
"""
问题2解决方案演示：基于历史功率的光伏电站日前发电功率预测

这是一个简化版本，展示核心预测逻辑，避免复杂的依赖问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 简化的中文字体设置
try:
    import matplotlib as mpl
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    mpl.font_manager.fontManager.addfont(font_path)  
    mpl.rc('font', family='simhei')
except:
    print("⚠️ 中文字体设置失败，将使用默认字体")

warnings.filterwarnings('ignore')

class SimplePowerForecaster:
    """简化版光伏功率预测器"""
    
    def __init__(self, station_id: str, data_dir: str = "../PVODdatasets_v1.0"):
        self.station_id = station_id
        self.data_dir = Path(data_dir)
        self.models = {}
        self.scalers = {}
        self.forecast_horizon = 7 * 24 * 4  # 7天 * 24小时 * 4个15分钟
        
        # 创建输出目录
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🚀 初始化 {station_id} 简化预测模型")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """加载和预处理数据"""
        print(f"📊 加载 {self.station_id} 数据...")
        
        # 加载数据
        file_path = self.data_dir / f"{self.station_id}.csv"
        df = pd.read_csv(file_path)
        
        # 时间处理
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        
        # 数据清理
        df['power'] = pd.to_numeric(df['power'], errors='coerce')
        df['power'] = df['power'].fillna(0)
        
        # 创建时间特征
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        df['day_of_year'] = df['date_time'].dt.dayofyear
        
        # 创建滞后特征
        for lag in [1, 4, 24, 96]:  # 15分钟、1小时、6小时、24小时前
            df[f'power_lag_{lag}'] = df['power'].shift(lag)
        
        # 创建滑动窗口统计特征
        for window in [4, 12, 24, 96]:  # 1小时、3小时、6小时、24小时窗口
            df[f'power_mean_{window}'] = df['power'].rolling(window=window).mean()
            df[f'power_std_{window}'] = df['power'].rolling(window=window).std()
        
        # 删除包含NaN的行
        df = df.dropna().reset_index(drop=True)
        
        print(f"✅ 数据预处理完成，共 {len(df)} 条记录")
        print(f"📅 时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        """创建特征矩阵"""
        feature_cols = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'power_lag_1', 'power_lag_4', 'power_lag_24', 'power_lag_96',
            'power_mean_4', 'power_mean_12', 'power_mean_24', 'power_mean_96',
            'power_std_4', 'power_std_12', 'power_std_24', 'power_std_96'
        ]
        
        # 只使用存在的列
        available_cols = [col for col in feature_cols if col in df.columns]
        return df[available_cols].values
    
    def train_models(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """训练预测模型"""
        print(f"\n🎯 开始训练预测模型...")
        
        # 分割数据
        split_idx = int(len(df) * train_ratio)
        train_df = df[:split_idx].copy()
        test_df = df[split_idx:].copy()
        
        print(f"📊 训练集: {len(train_df)} 条记录")
        print(f"📊 测试集: {len(test_df)} 条记录")
        
        # 保存数据分割信息
        self.train_df = train_df
        self.test_df = test_df
        
        # 准备训练数据
        X_train = self.create_features(train_df)
        y_train = train_df['power'].values
        
        # 数据标准化
        self.scalers['X'] = MinMaxScaler()
        self.scalers['y'] = MinMaxScaler()
        
        X_train_scaled = self.scalers['X'].fit_transform(X_train)
        y_train_scaled = self.scalers['y'].fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # 训练随机森林模型
        print("🔄 训练随机森林模型...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train_scaled, y_train_scaled)
        
        # 训练线性回归模型
        print("🔄 训练线性回归模型...")
        self.models['lr'] = LinearRegression()
        self.models['lr'].fit(X_train_scaled, y_train_scaled)
        
        # 简单移动平均模型
        print("🔄 创建移动平均模型...")
        self.models['ma'] = {
            'window': 96,  # 24小时窗口
            'data': train_df['power'].values
        }
        
        print(f"✅ 模型训练完成，成功训练 {len(self.models)} 个模型")
    
    def predict_single_step(self, last_data: pd.DataFrame, model_name: str) -> float:
        """单步预测"""
        if model_name == 'ma':
            # 移动平均预测
            window = self.models['ma']['window']
            recent_data = self.models['ma']['data'][-window:]
            return np.mean(recent_data)
        
        # 机器学习模型预测
        X = self.create_features(last_data)
        if len(X) == 0:
            return 0.0
        
        X_scaled = self.scalers['X'].transform(X[-1:])
        y_pred_scaled = self.models[model_name].predict(X_scaled)[0]
        y_pred = self.scalers['y'].inverse_transform([[y_pred_scaled]])[0, 0]
        
        return max(y_pred, 0)  # 确保非负
    
    def multi_step_forecast(self, steps: int) -> dict:
        """多步预测"""
        print(f"\n🔮 开始7天预测 ({steps} 个时间点)...")
        
        predictions = {}
        
        # 获取最后的数据作为起点
        last_data = self.train_df.copy()
        
        for model_name in ['rf', 'lr', 'ma']:
            model_predictions = []
            current_data = last_data.copy()
            
            for step in range(steps):
                # 预测下一个时间点
                pred = self.predict_single_step(current_data, model_name)
                model_predictions.append(pred)
                
                # 更新数据（简化版本，只更新功率值）
                if len(current_data) > 0:
                    # 创建新的时间点
                    last_time = current_data['date_time'].iloc[-1]
                    new_time = last_time + timedelta(minutes=15)
                    
                    # 创建新行
                    new_row = current_data.iloc[-1:].copy()
                    new_row['date_time'] = new_time
                    new_row['power'] = pred
                    new_row['hour'] = new_time.hour
                    new_row['day_of_week'] = new_time.dayofweek
                    new_row['month'] = new_time.month
                    new_row['day_of_year'] = new_time.dayofyear
                    
                    # 更新滞后特征
                    for lag in [1, 4, 24, 96]:
                        if len(current_data) >= lag:
                            new_row[f'power_lag_{lag}'] = current_data['power'].iloc[-lag]
                    
                    # 添加到数据中
                    current_data = pd.concat([current_data, new_row], ignore_index=True)
                    
                    # 保持数据长度不要太长
                    if len(current_data) > 1000:
                        current_data = current_data.iloc[-500:].reset_index(drop=True)
            
            predictions[model_name] = np.array(model_predictions)
        
        # 集成预测
        valid_predictions = [pred for pred in predictions.values() if len(pred) > 0]
        if valid_predictions:
            predictions['ensemble'] = np.mean(valid_predictions, axis=0)
        
        print(f"✅ 预测完成，生成 {len(predictions)} 组预测结果")
        
        return predictions
    
    def evaluate_predictions(self, predictions: dict) -> dict:
        """评估预测结果"""
        if len(self.test_df) == 0:
            print("⚠️ 没有测试数据，跳过评估")
            return {}
        
        print(f"\n📊 评估预测结果...")
        
        # 获取测试集真实值
        test_steps = min(len(self.test_df), self.forecast_horizon)
        y_true = self.test_df['power'].values[:test_steps]
        
        evaluation_results = {}
        
        for model_name, y_pred in predictions.items():
            if len(y_pred) == 0:
                continue
            
            # 截取相同长度
            pred_steps = min(len(y_pred), test_steps)
            y_pred_eval = y_pred[:pred_steps]
            y_true_eval = y_true[:pred_steps]
            
            # 计算评估指标
            mae = mean_absolute_error(y_true_eval, y_pred_eval)
            mse = mean_squared_error(y_true_eval, y_pred_eval)
            rmse = np.sqrt(mse)
            
            # 计算MAPE
            mask = y_true_eval != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true_eval[mask] - y_pred_eval[mask]) / y_true_eval[mask])) * 100
            else:
                mape = float('inf')
            
            # 计算R²
            r2 = r2_score(y_true_eval, y_pred_eval)
            
            evaluation_results[model_name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
            print(f"{model_name:10} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        
        return evaluation_results
    
    def save_results(self, predictions: dict, evaluation: dict):
        """保存结果"""
        print(f"\n💾 保存预测结果...")
        
        # 创建时间索引
        last_time = self.train_df['date_time'].iloc[-1]
        future_times = pd.date_range(
            start=last_time + timedelta(minutes=15),
            periods=self.forecast_horizon,
            freq='15T'
        )
        
        # 保存预测结果
        results_df = pd.DataFrame({'date_time': future_times})
        
        for model_name, pred in predictions.items():
            if len(pred) > 0:
                results_df[f'predicted_power_{model_name}'] = pred[:len(future_times)]
        
        # 如果有测试数据，也保存真实值
        if len(self.test_df) > 0:
            test_steps = min(len(self.test_df), len(future_times))
            results_df.loc[:test_steps-1, 'actual_power'] = self.test_df['power'].values[:test_steps]
        
        # 保存到CSV
        output_file = self.output_dir / f"{self.station_id}_simple_forecast.csv"
        results_df.to_csv(output_file, index=False)
        print(f"✅ 预测结果已保存到: {output_file}")
        
        # 保存评估结果
        if evaluation:
            eval_df = pd.DataFrame(evaluation).T
            eval_file = self.output_dir / f"{self.station_id}_simple_evaluation.csv"
            eval_df.to_csv(eval_file)
            print(f"✅ 评估结果已保存到: {eval_file}")
        
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame):
        """绘制结果"""
        print(f"\n🎨 生成可视化图表...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'{self.station_id} 光伏功率7天预测结果', fontsize=16, fontweight='bold')
            
            # 获取预测列
            pred_cols = [col for col in results_df.columns if col.startswith('predicted_power_')]
            
            # 1. 时间序列预测图
            ax1 = axes[0, 0]
            
            # 绘制真实值（如果有）
            if 'actual_power' in results_df.columns:
                mask = results_df['actual_power'].notna()
                ax1.plot(results_df.loc[mask, 'date_time'], 
                        results_df.loc[mask, 'actual_power'], 
                        'k-', linewidth=2, label='实际功率', alpha=0.8)
            
            # 绘制各模型预测
            colors = ['red', 'blue', 'green', 'orange']
            for i, col in enumerate(pred_cols):
                model_name = col.replace('predicted_power_', '')
                ax1.plot(results_df['date_time'], results_df[col], 
                        color=colors[i % len(colors)], linewidth=1.5, 
                        label=f'{model_name}预测', alpha=0.7)
            
            ax1.set_title('7天功率预测时间序列')
            ax1.set_xlabel('时间')
            ax1.set_ylabel('功率 (MW)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 预测统计对比
            ax2 = axes[0, 1]
            
            stats_data = []
            for col in pred_cols:
                model_name = col.replace('predicted_power_', '')
                stats_data.append({
                    '模型': model_name,
                    '平均功率': results_df[col].mean(),
                    '最大功率': results_df[col].max(),
                    '标准差': results_df[col].std()
                })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                x_pos = np.arange(len(stats_df))
                width = 0.25
                
                ax2.bar(x_pos - width, stats_df['平均功率'], width, label='平均功率', alpha=0.7)
                ax2.bar(x_pos, stats_df['最大功率'], width, label='最大功率', alpha=0.7)
                ax2.bar(x_pos + width, stats_df['标准差'], width, label='标准差', alpha=0.7)
                
                ax2.set_title('模型预测统计对比')
                ax2.set_xlabel('模型')
                ax2.set_ylabel('功率 (MW)')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(stats_df['模型'])
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. 小时平均功率
            ax3 = axes[1, 0]
            
            results_df['hour'] = results_df['date_time'].dt.hour
            hourly_stats = results_df.groupby('hour').agg({
                col: 'mean' for col in pred_cols
            }).reset_index()
            
            for col in pred_cols:
                model_name = col.replace('predicted_power_', '')
                ax3.plot(hourly_stats['hour'], hourly_stats[col], 
                        marker='o', linewidth=2, label=f'{model_name}预测')
            
            ax3.set_title('小时平均功率变化')
            ax3.set_xlabel('小时')
            ax3.set_ylabel('平均功率 (MW)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(0, 24, 2))
            
            # 4. 功率分布
            ax4 = axes[1, 1]
            
            for col in pred_cols:
                model_name = col.replace('predicted_power_', '')
                ax4.hist(results_df[col], bins=20, alpha=0.6, label=f'{model_name}预测', density=True)
            
            ax4.set_title('功率分布对比')
            ax4.set_xlabel('功率 (MW)')
            ax4.set_ylabel('密度')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            output_file = self.output_dir / f"{self.station_id}_simple_forecast_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化图表已保存到: {output_file}")
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")


def main():
    """主函数"""
    print("🌞 简化版光伏电站日前发电功率预测系统")
    print("=" * 60)
    print("问题2解决方案演示：基于历史功率的7天预测模型")
    print("=" * 60)
    
    # 选择要分析的站点
    station_id = "station01"
    
    try:
        # 初始化预测器
        forecaster = SimplePowerForecaster(station_id)
        
        # 加载和预处理数据
        df = forecaster.load_and_preprocess_data()
        
        # 训练模型
        forecaster.train_models(df)
        
        # 进行7天预测
        predictions = forecaster.multi_step_forecast(forecaster.forecast_horizon)
        
        # 评估预测结果
        evaluation = forecaster.evaluate_predictions(predictions)
        
        # 保存结果
        results_df = forecaster.save_results(predictions, evaluation)
        
        # 生成可视化
        forecaster.plot_results(results_df)
        
        print(f"\n🎉 {station_id} 预测完成！")
        print(f"📁 结果已保存到: p2/results/")
        
        return forecaster, results_df
        
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    forecaster, results = main() 