# 光伏性能可视化分析模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime
import warnings

# 简化的中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')

warnings.filterwarnings('ignore')

class PerformanceVisualizer:
    """光伏性能可视化分析器"""
    
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
    
    def plot_power_comparison_overview(self, df: pd.DataFrame, station_id: str):
        """绘制功率对比概览图"""
        self.ensure_chinese_font()
        
        # 过滤白昼数据：太阳高度角>0 且 理论功率>0 且 实际功率>=0
        daytime_df = df[
            (df['solar_elevation'] > 0) & 
            (df['theoretical_power'] > 0) & 
            (df['power'] >= 0)
        ].copy()
        
        if len(daytime_df) == 0:
            print(f"⚠️ {station_id} 没有白天数据，跳过概览图")
            return
        
        print(f"📊 {station_id} 概览分析 - 白昼数据: {len(daytime_df):,} 条")
        
        # 创建2x2子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 实际vs理论功率散点图
        sample_size = min(5000, len(daytime_df))
        sample_df = daytime_df.sample(sample_size)
        
        scatter = ax1.scatter(sample_df['theoretical_power'], sample_df['power'], 
                            alpha=0.6, s=15, c=sample_df['solar_elevation'], cmap='viridis')
        ax1.plot([0, sample_df['theoretical_power'].max()], [0, sample_df['theoretical_power'].max()], 
                'r--', linewidth=2, label='理想线 (1:1)')
        ax1.set_xlabel('理论功率 (MW)', fontsize=12)
        ax1.set_ylabel('实际功率 (MW)', fontsize=12)
        ax1.set_title(f'{station_id} 实际功率 vs 理论功率', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 添加颜色条
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('太阳高度角 (°)', fontsize=10)
        
        # 2. 性能比分布直方图
        performance_ratios = daytime_df['performance_ratio']
        performance_ratios = performance_ratios[performance_ratios.between(0, 2)]  # 过滤异常值
        
        ax2.hist(performance_ratios, bins=50, alpha=0.8, color='orange', edgecolor='darkorange')
        ax2.axvline(performance_ratios.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'平均值: {performance_ratios.mean():.3f}')
        ax2.set_xlabel('性能比 (实际/理论)', fontsize=12)
        ax2.set_ylabel('频次', fontsize=12)
        ax2.set_title(f'{station_id} 性能比分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 功率差值时间序列
        time_sample = daytime_df.sample(min(2000, len(daytime_df))).sort_values('date_time')
        ax3.plot(time_sample['date_time'], time_sample['power_difference'], 
                alpha=0.7, linewidth=1, color='green')
        ax3.axhline(0, color='red', linestyle='--', linewidth=1)
        ax3.set_xlabel('时间', fontsize=12)
        ax3.set_ylabel('功率差值 (实际-理论) MW', fontsize=12)
        ax3.set_title(f'{station_id} 功率差值时间序列', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 格式化日期轴
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 太阳高度角vs性能比
        elevation_bins = pd.cut(sample_df['solar_elevation'], bins=10)
        elevation_performance = sample_df.groupby(elevation_bins)['performance_ratio'].mean()
        
        bin_centers = [interval.mid for interval in elevation_performance.index]
        ax4.bar(range(len(bin_centers)), elevation_performance.values, 
               alpha=0.8, color='skyblue', edgecolor='navy')
        ax4.set_xlabel('太阳高度角区间', fontsize=12)
        ax4.set_ylabel('平均性能比', fontsize=12)
        ax4.set_title(f'{station_id} 太阳高度角与性能关系', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 设置x轴标签
        ax4.set_xticks(range(len(bin_centers)))
        ax4.set_xticklabels([f'{c:.1f}°' for c in bin_centers], rotation=45)
        
        self.save_plot(fig, f'{station_id}_功率对比概览分析.png')
    
    def plot_seasonal_analysis(self, df: pd.DataFrame, station_id: str):
        """绘制季节性分析图"""
        self.ensure_chinese_font()
        
        # 过滤白昼数据：太阳高度角>0 且 理论功率>0 且 实际功率>=0
        daytime_df = df[
            (df['solar_elevation'] > 0) & 
            (df['theoretical_power'] > 0) & 
            (df['power'] >= 0)
        ].copy()
        
        if len(daytime_df) == 0:
            print(f"⚠️ {station_id} 没有白天数据，跳过季节性分析")
            return
        
        print(f"📊 {station_id} 季节性分析 - 白昼数据: {len(daytime_df):,} 条")
        
        # 创建2x2子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 季节性功率对比箱线图
        seasons = ['春季', '夏季', '秋季', '冬季']
        season_data_actual = [daytime_df[daytime_df['season'] == season]['power'] for season in seasons]
        season_data_theoretical = [daytime_df[daytime_df['season'] == season]['theoretical_power'] for season in seasons]
        
        # 实际功率箱线图
        bp1 = ax1.boxplot(season_data_actual, labels=seasons, patch_artist=True, 
                         positions=np.arange(1, len(seasons)+1) - 0.2, widths=0.3)
        # 理论功率箱线图
        bp2 = ax1.boxplot(season_data_theoretical, labels=seasons, patch_artist=True,
                         positions=np.arange(1, len(seasons)+1) + 0.2, widths=0.3)
        
        # 设置颜色
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('季节', fontsize=12)
        ax1.set_ylabel('功率 (MW)', fontsize=12)
        ax1.set_title(f'{station_id} 季节性功率分布对比', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend([bp1['boxes'][0], bp2['boxes'][0]], ['实际功率', '理论功率'])
        
        # 2. 季节性性能比
        seasonal_performance = daytime_df.groupby('season')['performance_ratio'].agg(['mean', 'std'])
        seasonal_performance = seasonal_performance.reindex(seasons)
        
        bars = ax2.bar(seasons, seasonal_performance['mean'], 
                      yerr=seasonal_performance['std'], capsize=5,
                      alpha=0.8, color=['lightgreen', 'gold', 'orange', 'lightblue'],
                      edgecolor='black')
        ax2.set_xlabel('季节', fontsize=12)
        ax2.set_ylabel('平均性能比', fontsize=12)
        ax2.set_title(f'{station_id} 季节性性能比变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, mean_val in zip(bars, seasonal_performance['mean']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 月度趋势分析
        monthly_stats = daytime_df.groupby('month').agg({
            'power': 'mean',
            'theoretical_power': 'mean',
            'performance_ratio': 'mean'
        })
        
        months = range(1, 13)
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                      '7月', '8月', '9月', '10月', '11月', '12月']
        
        ax3_twin = ax3.twinx()
        
        # 功率趋势
        line1 = ax3.plot(monthly_stats.index, monthly_stats['power'], 
                        'o-', linewidth=2, markersize=6, color='blue', label='实际功率')
        line2 = ax3.plot(monthly_stats.index, monthly_stats['theoretical_power'], 
                        's-', linewidth=2, markersize=6, color='red', label='理论功率')
        
        # 性能比趋势
        line3 = ax3_twin.plot(monthly_stats.index, monthly_stats['performance_ratio'], 
                             '^-', linewidth=2, markersize=6, color='green', label='性能比')
        
        ax3.set_xlabel('月份', fontsize=12)
        ax3.set_ylabel('功率 (MW)', fontsize=12, color='black')
        ax3_twin.set_ylabel('性能比', fontsize=12, color='green')
        ax3.set_title(f'{station_id} 月度功率与性能趋势', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0.5, 12.5)
        
        # 设置x轴标签
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels([month_names[i-1] for i in range(1, 13)], rotation=45)
        
        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # 4. 辐照度对比分析
        irradiance_comparison = daytime_df.groupby('season').agg({
            'nwp_globalirrad': 'mean',
            'theoretical_ghi': 'mean',
            'theoretical_poa': 'mean'
        })
        irradiance_comparison = irradiance_comparison.reindex(seasons)
        
        x_pos = np.arange(len(seasons))
        width = 0.25
        
        bars1 = ax4.bar(x_pos - width, irradiance_comparison['nwp_globalirrad'], 
                       width, label='NWP实测GHI', alpha=0.8, color='skyblue')
        bars2 = ax4.bar(x_pos, irradiance_comparison['theoretical_ghi'], 
                       width, label='理论GHI', alpha=0.8, color='orange')
        bars3 = ax4.bar(x_pos + width, irradiance_comparison['theoretical_poa'], 
                       width, label='理论POA', alpha=0.8, color='lightgreen')
        
        ax4.set_xlabel('季节', fontsize=12)
        ax4.set_ylabel('辐照度 (W/m²)', fontsize=12)
        ax4.set_title(f'{station_id} 季节性辐照度对比', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(seasons)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.save_plot(fig, f'{station_id}_季节性分析.png')
    
    def plot_daily_patterns(self, df: pd.DataFrame, station_id: str):
        """绘制日内变化模式分析"""
        self.ensure_chinese_font()
        
        # 过滤白昼数据：太阳高度角>0 且 理论功率>0 且 实际功率>=0
        daytime_df = df[
            (df['solar_elevation'] > 0) & 
            (df['theoretical_power'] > 0) & 
            (df['power'] >= 0)
        ].copy()
        
        if len(daytime_df) == 0:
            print(f"⚠️ {station_id} 没有白天数据，跳过日内分析")
            return
        
        print(f"📊 {station_id} 日内分析 - 白昼数据: {len(daytime_df):,} 条")
        
        # 创建2x2子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 日内功率变化曲线
        hourly_stats = daytime_df.groupby('hour').agg({
            'power': ['mean', 'std'],
            'theoretical_power': ['mean', 'std'],
            'performance_ratio': ['mean', 'std']
        })
        
        hours = hourly_stats.index
        
        # 实际功率曲线
        ax1.plot(hours, hourly_stats['power']['mean'], 'o-', linewidth=2, 
                markersize=6, color='blue', label='实际功率')
        ax1.fill_between(hours, 
                        hourly_stats['power']['mean'] - hourly_stats['power']['std'],
                        hourly_stats['power']['mean'] + hourly_stats['power']['std'],
                        alpha=0.3, color='blue')
        
        # 理论功率曲线
        ax1.plot(hours, hourly_stats['theoretical_power']['mean'], 's-', linewidth=2,
                markersize=6, color='red', label='理论功率')
        ax1.fill_between(hours,
                        hourly_stats['theoretical_power']['mean'] - hourly_stats['theoretical_power']['std'],
                        hourly_stats['theoretical_power']['mean'] + hourly_stats['theoretical_power']['std'],
                        alpha=0.3, color='red')
        
        ax1.set_xlabel('小时', fontsize=12)
        ax1.set_ylabel('功率 (MW)', fontsize=12)
        ax1.set_title(f'{station_id} 日内功率变化模式', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(5, 19)  # 只显示白天时段
        
        # 2. 日内性能比变化
        ax2.plot(hours, hourly_stats['performance_ratio']['mean'], '^-', linewidth=2,
                markersize=6, color='green', label='平均性能比')
        ax2.fill_between(hours,
                        hourly_stats['performance_ratio']['mean'] - hourly_stats['performance_ratio']['std'],
                        hourly_stats['performance_ratio']['mean'] + hourly_stats['performance_ratio']['std'],
                        alpha=0.3, color='green')
        
        ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, label='理想性能比')
        ax2.set_xlabel('小时', fontsize=12)
        ax2.set_ylabel('性能比', fontsize=12)
        ax2.set_title(f'{station_id} 日内性能比变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(5, 19)
        
        # 3. 太阳高度角与功率关系热力图
        # 创建小时-月份的功率热力图
        pivot_data = daytime_df.pivot_table(values='performance_ratio', 
                                           index='hour', columns='month', 
                                           aggfunc='mean')
        
        im = ax3.imshow(pivot_data.values, aspect='auto', cmap='RdYlBu_r', 
                       origin='lower', interpolation='nearest')
        ax3.set_xlabel('月份', fontsize=12)
        ax3.set_ylabel('小时', fontsize=12)
        ax3.set_title(f'{station_id} 小时-月份性能比热力图', fontsize=14, fontweight='bold')
        
        # 设置坐标轴
        ax3.set_xticks(range(len(pivot_data.columns)))
        ax3.set_xticklabels([f'{m}月' for m in pivot_data.columns])
        ax3.set_yticks(range(len(pivot_data.index)))
        ax3.set_yticklabels([f'{h}:00' for h in pivot_data.index])
        
        # 添加颜色条
        cbar3 = plt.colorbar(im, ax=ax3)
        cbar3.set_label('性能比', fontsize=10)
        
        # 4. 功率损失分析
        daytime_df['power_loss'] = daytime_df['theoretical_power'] - daytime_df['power']
        daytime_df['power_loss_percent'] = (daytime_df['power_loss'] / daytime_df['theoretical_power']) * 100
        
        # 按太阳高度角分组分析功率损失
        elevation_bins = pd.cut(daytime_df['solar_elevation'], bins=8)
        loss_by_elevation = daytime_df.groupby(elevation_bins)['power_loss_percent'].mean()
        
        bin_centers = [interval.mid for interval in loss_by_elevation.index]
        bars = ax4.bar(range(len(bin_centers)), loss_by_elevation.values,
                      alpha=0.8, color='coral', edgecolor='darkred')
        
        ax4.set_xlabel('太阳高度角区间', fontsize=12)
        ax4.set_ylabel('平均功率损失 (%)', fontsize=12)
        ax4.set_title(f'{station_id} 太阳高度角与功率损失关系', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 设置x轴标签
        ax4.set_xticks(range(len(bin_centers)))
        ax4.set_xticklabels([f'{c:.1f}°' for c in bin_centers], rotation=45)
        
        # 添加数值标签
        for bar, val in zip(bars, loss_by_elevation.values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        self.save_plot(fig, f'{station_id}_日内变化分析.png')
    
    def plot_weather_impact_analysis(self, df: pd.DataFrame, station_id: str):
        """绘制天气影响分析"""
        self.ensure_chinese_font()
        
        # 过滤白昼数据：太阳高度角>0 且 理论功率>0 且 实际功率>=0
        daytime_df = df[
            (df['solar_elevation'] > 0) & 
            (df['theoretical_power'] > 0) & 
            (df['power'] >= 0)
        ].copy()
        
        if len(daytime_df) == 0:
            print(f"⚠️ {station_id} 没有白天数据，跳过天气影响分析")
            return
        
        print(f"📊 {station_id} 天气影响分析 - 白昼数据: {len(daytime_df):,} 条")
        
        # 创建2x2子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 温度对性能的影响
        temp_bins = pd.cut(daytime_df['nwp_temperature'], bins=10)
        temp_performance = daytime_df.groupby(temp_bins)['performance_ratio'].mean()
        
        bin_centers = [interval.mid for interval in temp_performance.index if not pd.isna(interval.mid)]
        performance_values = [temp_performance[interval] for interval in temp_performance.index if not pd.isna(interval.mid)]
        
        ax1.plot(bin_centers, performance_values, 'o-', linewidth=2, markersize=6, color='red')
        ax1.set_xlabel('环境温度 (°C)', fontsize=12)
        ax1.set_ylabel('平均性能比', fontsize=12)
        ax1.set_title(f'{station_id} 温度对性能的影响', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 湿度对性能的影响
        humidity_bins = pd.cut(daytime_df['nwp_humidity'], bins=10)
        humidity_performance = daytime_df.groupby(humidity_bins)['performance_ratio'].mean()
        
        bin_centers_hum = [interval.mid for interval in humidity_performance.index if not pd.isna(interval.mid)]
        performance_values_hum = [humidity_performance[interval] for interval in humidity_performance.index if not pd.isna(interval.mid)]
        
        ax2.plot(bin_centers_hum, performance_values_hum, 's-', linewidth=2, markersize=6, color='blue')
        ax2.set_xlabel('相对湿度 (%)', fontsize=12)
        ax2.set_ylabel('平均性能比', fontsize=12)
        ax2.set_title(f'{station_id} 湿度对性能的影响', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 实测辐照度vs理论辐照度对比
        sample_size = min(3000, len(daytime_df))
        sample_df = daytime_df.sample(sample_size)
        
        scatter = ax3.scatter(sample_df['theoretical_ghi'], sample_df['nwp_globalirrad'],
                            alpha=0.6, s=15, c=sample_df['performance_ratio'], cmap='RdYlGn')
        ax3.plot([0, sample_df['theoretical_ghi'].max()], [0, sample_df['theoretical_ghi'].max()],
                'r--', linewidth=2, label='理想线 (1:1)')
        ax3.set_xlabel('理论全球水平辐照度 (W/m²)', fontsize=12)
        ax3.set_ylabel('NWP实测全球辐照度 (W/m²)', fontsize=12)
        ax3.set_title(f'{station_id} 理论vs实测辐照度对比', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 添加颜色条
        cbar3 = plt.colorbar(scatter, ax=ax3)
        cbar3.set_label('性能比', fontsize=10)
        
        # 4. 风速对性能的影响
        windspeed_bins = pd.cut(daytime_df['nwp_windspeed'], bins=8)
        windspeed_performance = daytime_df.groupby(windspeed_bins)['performance_ratio'].mean()
        
        bin_centers_wind = [interval.mid for interval in windspeed_performance.index if not pd.isna(interval.mid)]
        performance_values_wind = [windspeed_performance[interval] for interval in windspeed_performance.index if not pd.isna(interval.mid)]
        
        bars = ax4.bar(range(len(bin_centers_wind)), performance_values_wind,
                      alpha=0.8, color='lightgreen', edgecolor='darkgreen')
        ax4.set_xlabel('风速区间', fontsize=12)
        ax4.set_ylabel('平均性能比', fontsize=12)
        ax4.set_title(f'{station_id} 风速对性能的影响', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 设置x轴标签
        ax4.set_xticks(range(len(bin_centers_wind)))
        ax4.set_xticklabels([f'{c:.1f}' for c in bin_centers_wind], rotation=45)
        
        # 添加数值标签
        for bar, val in zip(bars, performance_values_wind):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        self.save_plot(fig, f'{station_id}_天气影响分析.png')
    
    def create_comprehensive_report(self, df: pd.DataFrame, station_id: str, stats: dict):
        """创建综合分析报告"""
        print(f"\n📊 正在生成 {station_id} 的综合可视化分析...")
        
        # 生成所有分析图表
        self.plot_power_comparison_overview(df, station_id)
        self.plot_seasonal_analysis(df, station_id)
        self.plot_daily_patterns(df, station_id)
        self.plot_weather_impact_analysis(df, station_id)
        
        print(f"✅ {station_id} 可视化分析完成！")
        print(f"📁 所有图表已保存到: {self.output_dir}")


def main():
    """主函数 - 演示可视化功能"""
    # 这里可以加载之前计算的结果进行可视化
    results_dir = Path("results")
    
    # 检查是否有计算结果
    csv_files = list(results_dir.glob("*_theoretical_vs_actual.csv"))
    
    if not csv_files:
        print("❌ 未找到理论vs实际分析结果文件")
        print("请先运行 solar_theoretical_model.py 生成分析数据")
        return
    
    # 创建可视化器
    visualizer = PerformanceVisualizer()
    
    # 对每个结果文件进行可视化
    for csv_file in csv_files:
        station_id = csv_file.stem.replace("_theoretical_vs_actual", "")
        print(f"\n🎨 正在为 {station_id} 创建可视化分析...")
        
        # 加载数据
        df = pd.read_csv(csv_file)
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        # 创建模拟统计数据
        daytime_df = df[df['theoretical_power'] > 0]
        stats = {
            'station_id': station_id,
            'mean_actual_power': daytime_df['power'].mean(),
            'mean_theoretical_power': daytime_df['theoretical_power'].mean(),
            'mean_performance_ratio': daytime_df['performance_ratio'].mean(),
        }
        
        # 生成综合报告
        visualizer.create_comprehensive_report(df, station_id, stats)


if __name__ == "__main__":
    main() 