# 预测结果可视化分析模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

# 简化的中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')

warnings.filterwarnings('ignore')

class PredictionVisualizer:
    """预测结果可视化分析器"""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['axes.unicode_minus'] = False
        
    def ensure_chinese_font(self):
        """确保中文字体设置正确应用"""
        mpl.rc('font', family='simhei')
        plt.rcParams['axes.unicode_minus'] = False
        
    def save_plot(self, fig, filename, title=""):
        """保存图表"""
        self.ensure_chinese_font()
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ 已保存: {filename}")
    
    def plot_model_performance(self, results: dict, station_id: str):
        """绘制模型性能分析图"""
        self.ensure_chinese_font()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 预测vs实际值散点图
        y_test = results['y_test']
        y_pred = results['y_test_pred']
        
        ax1.scatter(y_test, y_pred, alpha=0.6, s=20)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax1.set_xlabel('实际功率 (MW)', fontsize=12)
        ax1.set_ylabel('预测功率 (MW)', fontsize=12)
        ax1.set_title(f'{station_id} 预测vs实际功率散点图', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 添加R²值
        r2 = results['test_metrics']['r2']
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        # 2. 时间序列对比
        test_dates = results['test_dates']
        sample_size = min(1000, len(y_test))  # 限制显示点数
        indices = np.linspace(0, len(y_test)-1, sample_size, dtype=int)
        
        ax2.plot(test_dates.iloc[indices], y_test.iloc[indices], 'b-', linewidth=1, label='实际功率', alpha=0.8)
        ax2.plot(test_dates.iloc[indices], y_pred[indices], 'r-', linewidth=1, label='预测功率', alpha=0.8)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('功率 (MW)', fontsize=12)
        ax2.set_title(f'{station_id} 测试集预测效果时间序列', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 格式化日期轴
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 残差分析
        residuals = y_test - y_pred
        ax3.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('预测功率 (MW)', fontsize=12)
        ax3.set_ylabel('残差 (实际-预测)', fontsize=12)
        ax3.set_title(f'{station_id} 残差分析', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 误差分布直方图
        ax4.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='navy')
        ax4.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {residuals.mean():.4f}')
        ax4.axvline(residuals.median(), color='green', linestyle='--', linewidth=2,
                   label=f'中位数: {residuals.median():.4f}')
        ax4.set_xlabel('残差 (MW)', fontsize=12)
        ax4.set_ylabel('频次', fontsize=12)
        ax4.set_title(f'{station_id} 残差分布', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.save_plot(fig, f'{station_id}_模型性能分析.png')
    
    def plot_forecast_results(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame, 
                            station_id: str, days_to_show: int = 14):
        """绘制预测结果图"""
        self.ensure_chinese_font()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 获取最后几天的历史数据用于对比
        last_date = historical_df['date_time'].max()
        start_date = last_date - timedelta(days=days_to_show-7)
        recent_data = historical_df[historical_df['date_time'] >= start_date].copy()
        
        # 1. 整体预测结果
        ax1.plot(recent_data['date_time'], recent_data['power'], 'b-', linewidth=2, 
                label='历史功率', alpha=0.8)
        ax1.plot(forecast_df['date_time'], forecast_df['predicted_power'], 'r-', linewidth=2,
                label='预测功率', alpha=0.8)
        
        # 添加分界线
        ax1.axvline(last_date, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label='预测起始点')
        
        ax1.set_xlabel('时间', fontsize=12)
        ax1.set_ylabel('功率 (MW)', fontsize=12)
        ax1.set_title(f'{station_id} 7天功率预测结果', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 格式化日期轴
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 日内模式对比
        # 计算历史数据的日内平均模式
        recent_data['hour_minute'] = recent_data['date_time'].dt.hour + recent_data['date_time'].dt.minute/60
        historical_pattern = recent_data.groupby('hour_minute')['power'].mean()
        
        # 计算预测数据的日内平均模式
        forecast_df['hour_minute'] = forecast_df['date_time'].dt.hour + forecast_df['date_time'].dt.minute/60
        forecast_pattern = forecast_df.groupby('hour_minute')['predicted_power'].mean()
        
        ax2.plot(historical_pattern.index, historical_pattern.values, 'b-', linewidth=2,
                label='历史日内平均模式', alpha=0.8)
        ax2.plot(forecast_pattern.index, forecast_pattern.values, 'r-', linewidth=2,
                label='预测日内平均模式', alpha=0.8)
        
        ax2.set_xlabel('时间 (小时)', fontsize=12)
        ax2.set_ylabel('平均功率 (MW)', fontsize=12)
        ax2.set_title(f'{station_id} 日内功率模式对比', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        
        self.save_plot(fig, f'{station_id}_7天预测结果.png')
    
    def plot_feature_importance(self, model, feature_names: list, station_id: str, top_n: int = 20):
        """绘制特征重要性图"""
        self.ensure_chinese_font()
        
        # 获取特征重要性
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 选择前N个重要特征
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color='skyblue', edgecolor='navy', alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('特征重要性', fontsize=12)
        ax.set_title(f'{station_id} XGBoost模型特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        self.save_plot(fig, f'{station_id}_特征重要性分析.png')
    
    def plot_daily_forecast_breakdown(self, forecast_df: pd.DataFrame, station_id: str):
        """绘制每日预测分解图"""
        self.ensure_chinese_font()
        
        # 按天分组
        forecast_df['date'] = forecast_df['date_time'].dt.date
        forecast_df['hour_minute'] = forecast_df['date_time'].dt.hour + forecast_df['date_time'].dt.minute/60
        
        unique_dates = sorted(forecast_df['date'].unique())
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_dates)))
        
        for i, date in enumerate(unique_dates):
            if i >= 8:  # 最多显示8天
                break
                
            day_data = forecast_df[forecast_df['date'] == date]
            
            axes[i].plot(day_data['hour_minute'], day_data['predicted_power'], 
                        color=colors[i], linewidth=2, marker='o', markersize=3)
            axes[i].set_title(f'{date}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('时间 (小时)', fontsize=10)
            axes[i].set_ylabel('预测功率 (MW)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 24)
            
            # 添加统计信息
            daily_max = day_data['predicted_power'].max()
            daily_mean = day_data['predicted_power'].mean()
            axes[i].text(0.02, 0.98, f'峰值: {daily_max:.2f}MW\n平均: {daily_mean:.2f}MW', 
                        transform=axes[i].transAxes, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=9)
        
        # 隐藏多余的子图
        for i in range(len(unique_dates), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{station_id} 7天逐日功率预测分解', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_plot(fig, f'{station_id}_逐日预测分解.png')
    
    def create_comprehensive_report(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame,
                                  results: dict, model, feature_names: list, station_id: str):
        """创建综合预测分析报告"""
        print(f"\n📊 正在生成 {station_id} 的综合预测分析...")
        
        # 生成所有分析图表
        self.plot_model_performance(results, station_id)
        self.plot_forecast_results(historical_df, forecast_df, station_id)
        self.plot_feature_importance(model, feature_names, station_id)
        self.plot_daily_forecast_breakdown(forecast_df, station_id)
        
        print(f"✅ {station_id} 预测分析可视化完成！")
        print(f"📁 所有图表已保存到: {self.output_dir}")


def main():
    """主函数 - 演示可视化功能"""
    # 这里可以加载之前的预测结果进行可视化
    results_dir = Path("results")
    
    # 检查是否有预测结果
    forecast_files = list(results_dir.glob("*_7day_forecast.csv"))
    
    if not forecast_files:
        print("❌ 未找到预测结果文件")
        print("请先运行 power_prediction_model.py 生成预测数据")
        return
    
    # 创建可视化器
    visualizer = PredictionVisualizer()
    
    print(f"🎨 找到 {len(forecast_files)} 个预测结果文件")
    
    for forecast_file in forecast_files:
        station_id = forecast_file.stem.replace("_7day_forecast", "")
        print(f"\n🎨 正在为 {station_id} 创建可视化分析...")
        
        # 加载预测结果
        forecast_df = pd.read_csv(forecast_file)
        forecast_df['date_time'] = pd.to_datetime(forecast_df['date_time'])
        
        print(f"📊 预测数据: {len(forecast_df)} 个时间点")
        print(f"📅 预测时间范围: {forecast_df['date_time'].min()} 到 {forecast_df['date_time'].max()}")


if __name__ == "__main__":
    main() 