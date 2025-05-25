# PVOD数据集分析器 (修正版 - 简化中文字体设置)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import matplotlib.dates as mdates
import matplotlib

# 简化的中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')

# 设置警告
warnings.filterwarnings('ignore')

class PVODAnalyzerCorrectedFixed:
    """PVOD光伏数据集分析器 (修正版 - 简化中文字体设置)"""
    
    def __init__(self, data_dir: str = "PVODdatasets_v1.0", figures_dir: str = "Figures"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # 设置基本图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['axes.unicode_minus'] = False
        
        # 数据列映射
        self.column_mapping = {
            'date_time': '时间',
            'nwp_globalirrad': 'NWP全球辐射',
            'nwp_directirrad': 'NWP直射辐射',
            'nwp_temperature': 'NWP温度',
            'nwp_humidity': 'NWP湿度',
            'nwp_windspeed': 'NWP风速',
            'nwp_winddirection': 'NWP风向',
            'nwp_pressure': 'NWP气压',
            'lmd_totalirrad': 'LMD总辐射',
            'lmd_diffuseirrad': 'LMD散射辐射',
            'lmd_temperature': 'LMD温度',
            'lmd_pressure': 'LMD气压',
            'lmd_winddirection': 'LMD风向',
            'lmd_windspeed': 'LMD风速',
            'power': '功率(MW)'
        }
    
    def test_chinese_display(self):
        """测试中文显示效果"""
        print("🧪 测试中文字体显示...")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # 测试数据
        test_data = [20, 18, 16, 15, 14]
        test_labels = ['station00', 'station01', 'station02', 'station03', 'station04']
        
        bars = ax.bar(test_labels, test_data, color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db'])
        ax.set_title('PVOD光伏站点容量因子测试图', fontsize=14, fontweight='bold')
        ax.set_xlabel('光伏站点编号')
        ax.set_ylabel('容量因子 (%)')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, test_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                   f'{val}%', ha='center', fontweight='bold')
        
        # 添加说明文字
        ax.text(0.5, 0.95, '如果您能看到这些中文字符，说明字体配置成功！', 
               transform=ax.transAxes, ha='center', va='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'chinese_font_test_simplified.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✓ 中文字体测试图已保存: Figures/chinese_font_test_simplified.png")
        
    def load_metadata(self) -> pd.DataFrame:
        """加载元数据"""
        metadata_path = self.data_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        return metadata
    
    def load_station_data(self, station_id: str) -> pd.DataFrame:
        """加载单个站点数据"""
        file_path = self.data_dir / f"{station_id}.csv"
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # 数据清理
            df = self._clean_data(df)
            
            # 时间处理
            df = self._process_time_data(df)
            
            return df
        except Exception as e:
            print(f"加载 {station_id} 数据失败: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        # 处理时间列
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        
        # 转换数值列
        numeric_columns = [col for col in df.columns if col != 'date_time']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除异常值
        for col in numeric_columns:
            if col == 'power':
                # 功率不能为负值
                df[col] = df[col].clip(lower=0)
            elif 'irrad' in col:
                # 辐射值不能为负
                df[col] = df[col].clip(lower=0)
            elif 'temperature' in col:
                # 温度范围限制
                df[col] = df[col].clip(-50, 60)
            elif 'humidity' in col:
                # 湿度范围限制
                df[col] = df[col].clip(0, 100)
            elif 'windspeed' in col:
                # 风速不能为负
                df[col] = df[col].clip(lower=0)
        
        return df
    
    def _process_time_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理时间数据"""
        if 'date_time' in df.columns:
            df['hour'] = df['date_time'].dt.hour
            df['day'] = df['date_time'].dt.day
            df['month'] = df['date_time'].dt.month
            df['year'] = df['date_time'].dt.year
            df['season'] = df['month'].map({
                12: '冬季', 1: '冬季', 2: '冬季',
                3: '春季', 4: '春季', 5: '春季',
                6: '夏季', 7: '夏季', 8: '夏季',
                9: '秋季', 10: '秋季', 11: '秋季'
            })
            df['weekday'] = df['date_time'].dt.day_name()
            
            # 添加白天/夜晚标记
            df['is_daytime'] = (df['hour'] >= 6) & (df['hour'] <= 18)
            
        return df
    
    def load_all_stations(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """加载所有站点数据"""
        metadata = self.load_metadata()
        station_data = {}
        
        for _, row in metadata.iterrows():
            station_id = row['Station_ID']
            print(f"正在加载 {station_id}...")
            
            df = self.load_station_data(station_id)
            if not df.empty:
                # 添加容量信息 - 注意: metadata中容量单位是kW，功率数据单位是MW
                capacity_kw = row['Capacity']
                capacity_mw = capacity_kw / 1000  # 转换为MW
                
                df['capacity_kw'] = capacity_kw
                df['capacity_mw'] = capacity_mw
                # 修正容量因子计算：功率数据已经是MW单位
                df['capacity_factor'] = (df['power'] / capacity_mw) * 100
                df['technology'] = row['PV_Technology']
                df['longitude'] = row['Longitude']
                df['latitude'] = row['Latitude']
                
                station_data[station_id] = df
                print(f"{station_id} 数据加载成功，形状: {df.shape}")
        
        return station_data, metadata
    
    def create_overview_analysis(self, station_data: Dict, metadata: pd.DataFrame):
        """创建概览分析"""
        print("创建数据概览分析...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('PVOD光伏数据集概览分析', fontsize=16, fontweight='bold')
        
        # 1. 各站点装机容量
        capacities = metadata['Capacity'].values / 1000  # 转换为MW
        station_names = metadata['Station_ID'].values
        
        bars = axes[0, 0].bar(range(len(station_names)), capacities, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('各站点装机容量', fontweight='bold')
        axes[0, 0].set_xlabel('站点编号')
        axes[0, 0].set_ylabel('装机容量 (MW)')
        axes[0, 0].set_xticks(range(len(station_names)))
        axes[0, 0].set_xticklabels(station_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 在柱状图上添加数值
        for bar, cap in zip(bars, capacities):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{cap:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 地理分布
        lons = metadata['Longitude'].values
        lats = metadata['Latitude'].values
        scatter = axes[0, 1].scatter(lons, lats, s=capacities*10, alpha=0.6, c=capacities, cmap='viridis')
        axes[0, 1].set_title('站点地理分布', fontweight='bold')
        axes[0, 1].set_xlabel('经度 (°)')
        axes[0, 1].set_ylabel('纬度 (°)')
        axes[0, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[0, 1])
        cbar.set_label('装机容量 (MW)')
        
        # 添加站点标签
        for i, name in enumerate(station_names):
            axes[0, 1].annotate(name, (lons[i], lats[i]), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        
        # 3. 技术类型分布
        tech_counts = metadata['PV_Technology'].value_counts()
        colors = ['lightblue', 'lightcoral']
        wedges, texts, autotexts = axes[0, 2].pie(tech_counts.values, labels=tech_counts.index, 
                                                 autopct='%1.1f%%', colors=colors)
        axes[0, 2].set_title('光伏技术类型分布', fontweight='bold')
        
        # 4. 数据时间跨度
        time_spans = []
        data_counts = []
        for station_id, df in station_data.items():
            if 'date_time' in df.columns:
                time_span = (df['date_time'].max() - df['date_time'].min()).days
                time_spans.append(time_span)
                data_counts.append(len(df))
        
        if time_spans:
            bars = axes[1, 0].bar(range(len(time_spans)), time_spans, alpha=0.7, color='orange')
            axes[1, 0].set_title('各站点数据时间跨度', fontweight='bold')
            axes[1, 0].set_xlabel('站点编号')
            axes[1, 0].set_ylabel('数据跨度 (天)')
            axes[1, 0].set_xticks(range(len(station_names)))
            axes[1, 0].set_xticklabels(station_names, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, span in zip(bars, time_spans):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 5,
                               f'{span}天', ha='center', va='bottom', fontsize=8)
        
        # 5. 数据量分布
        if data_counts:
            bars = axes[1, 1].bar(range(len(data_counts)), data_counts, alpha=0.7, color='green')
            axes[1, 1].set_title('各站点数据记录量', fontweight='bold')
            axes[1, 1].set_xlabel('站点编号')
            axes[1, 1].set_ylabel('数据记录数')
            axes[1, 1].set_xticks(range(len(station_names)))
            axes[1, 1].set_xticklabels(station_names, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, data_counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 500,
                               f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # 6. 各站点平均容量因子
        avg_cf = []
        for station_id, df in station_data.items():
            avg_cf.append(df['capacity_factor'].mean())
        
        if avg_cf:
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(avg_cf)))
            bars = axes[1, 2].bar(range(len(station_names)), avg_cf, alpha=0.8, color=colors)
            axes[1, 2].set_title('各站点平均容量因子', fontweight='bold')
            axes[1, 2].set_xlabel('站点编号')
            axes[1, 2].set_ylabel('平均容量因子 (%)')
            axes[1, 2].set_xticks(range(len(station_names)))
            axes[1, 2].set_xticklabels(station_names, rotation=45)
            axes[1, 2].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, cf in zip(bars, avg_cf):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{cf:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pvod_overview_analysis_fixed.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ 概览分析图表已保存")
    
    def create_power_generation_analysis(self, station_data: Dict):
        """创建发电分析"""
        print("创建发电分析...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('PVOD光伏发电深度分析', fontsize=16, fontweight='bold')
        
        # 收集所有站点数据
        all_power = []
        all_cf = []
        all_hours = []
        all_months = []
        station_labels = []
        
        for station_id, df in station_data.items():
            power_data = df['power'].dropna()
            cf_data = df['capacity_factor'].dropna()
            
            all_power.extend(power_data)
            all_cf.extend(cf_data)
            station_labels.extend([station_id] * len(power_data))
            
            if 'hour' in df.columns:
                all_hours.extend(df['hour'])
            if 'month' in df.columns:
                all_months.extend(df['month'])
        
        # 1. 功率分布直方图
        axes[0, 0].hist(all_power, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('功率输出分布直方图', fontweight='bold')
        axes[0, 0].set_xlabel('功率输出 (MW)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_power = np.mean(all_power)
        max_power = np.max(all_power)
        axes[0, 0].axvline(mean_power, color='red', linestyle='--', 
                          label=f'平均值: {mean_power:.1f} MW')
        axes[0, 0].axvline(max_power, color='green', linestyle='--', 
                          label=f'最大值: {max_power:.1f} MW')
        axes[0, 0].legend()
        
        # 2. 容量因子分布
        axes[0, 1].hist(all_cf, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('容量因子分布直方图', fontweight='bold')
        axes[0, 1].set_xlabel('容量因子 (%)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_cf = np.mean(all_cf)
        axes[0, 1].axvline(mean_cf, color='red', linestyle='--', 
                          label=f'平均值: {mean_cf:.1f}%')
        axes[0, 1].legend()
        
        # 3. 日内发电模式
        if all_hours:
            hour_power = pd.DataFrame({'hour': all_hours, 'power': all_power[:len(all_hours)]})
            hourly_avg = hour_power.groupby('hour')['power'].mean()
            axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
            axes[1, 0].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
            axes[1, 0].set_title('24小时日内发电模式', fontweight='bold')
            axes[1, 0].set_xlabel('小时 (0-23)')
            axes[1, 0].set_ylabel('平均功率 (MW)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(0, 23)
            
            # 标注峰值时间
            peak_hour = hourly_avg.idxmax()
            peak_power = hourly_avg.max()
            axes[1, 0].annotate(f'峰值时间: {peak_hour}:00\n峰值功率: {peak_power:.1f} MW', 
                              xy=(peak_hour, peak_power), xytext=(peak_hour+2, peak_power+1),
                              arrowprops=dict(arrowstyle='->', color='red'),
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # 4. 月度发电模式
        if all_months:
            month_power = pd.DataFrame({'month': all_months, 'power': all_power[:len(all_months)]})
            monthly_avg = month_power.groupby('month')['power'].mean()
            month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                          '7月', '8月', '9月', '10月', '11月', '12月']
            
            bars = axes[1, 1].bar(monthly_avg.index, monthly_avg.values, alpha=0.7, color='green')
            axes[1, 1].set_title('月度发电模式分析', fontweight='bold')
            axes[1, 1].set_xlabel('月份')
            axes[1, 1].set_ylabel('平均功率 (MW)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(0.5, 12.5)
            
            # 设置x轴标签
            axes[1, 1].set_xticks(range(1, 13))
            axes[1, 1].set_xticklabels([month_names[i-1] for i in monthly_avg.index], rotation=45)
            
            # 在柱状图上添加数值
            for bar, month in zip(bars, monthly_avg.index):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 5. 各站点功率箱线图
        power_by_station = [station_data[station]['power'].dropna() for station in station_data.keys()]
        station_names = list(station_data.keys())
        
        box_plot = axes[2, 0].boxplot(power_by_station, labels=station_names, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[2, 0].set_title('各站点功率分布箱线图', fontweight='bold')
        axes[2, 0].set_xlabel('光伏站点')
        axes[2, 0].set_ylabel('功率输出 (MW)')
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 站点性能排名
        station_stats = []
        labels = []
        for station_id, df in station_data.items():
            mean_cf = df['capacity_factor'].mean()
            station_stats.append(mean_cf)
            labels.append(station_id)
        
        # 按容量因子排序
        sorted_data = sorted(zip(labels, station_stats), key=lambda x: x[1], reverse=True)
        sorted_labels, sorted_stats = zip(*sorted_data)
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_stats)))
        bars = axes[2, 1].bar(range(len(sorted_labels)), sorted_stats, color=colors, alpha=0.8)
        axes[2, 1].set_title('各站点平均容量因子性能排名', fontweight='bold')
        axes[2, 1].set_xlabel('站点 (按性能排序)')
        axes[2, 1].set_ylabel('平均容量因子 (%)')
        axes[2, 1].set_xticks(range(len(sorted_labels)))
        axes[2, 1].set_xticklabels(sorted_labels, rotation=45)
        axes[2, 1].grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签和排名
        for i, (bar, stat) in enumerate(zip(bars, sorted_stats)):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{stat:.1f}%', ha='center', va='bottom', fontsize=9)
            # 添加排名
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height/2,
                           f'#{i+1}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pvod_power_analysis_fixed.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ 发电分析图表已保存")
    
    def create_meteorological_correlation_analysis(self, station_data: Dict):
        """创建气象与发电相关性分析"""
        print("创建气象与发电相关性分析...")
        
        # 选择数据量最大的站点
        max_data_station = max(station_data.items(), key=lambda x: len(x[1]))
        station_id, df = max_data_station
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'气象条件与发电相关性深度分析 (基于{station_id})', fontsize=16, fontweight='bold')
        
        # 1. NWP全球辐射vs功率
        valid_indices = df['nwp_globalirrad'].notna() & df['power'].notna()
        if valid_indices.sum() > 1000:
            sample_size = min(5000, valid_indices.sum())
            sample_df = df[valid_indices].sample(sample_size)
            
            axes[0, 0].scatter(sample_df['nwp_globalirrad'], sample_df['power'], alpha=0.3, s=2)
            axes[0, 0].set_title('NWP全球辐射 vs 功率输出', fontweight='bold')
            axes[0, 0].set_xlabel('全球辐射强度 (W/m²)')
            axes[0, 0].set_ylabel('功率输出 (MW)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 计算相关系数
            corr = sample_df['nwp_globalirrad'].corr(sample_df['power'])
            axes[0, 0].text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=axes[0, 0].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 2. LMD总辐射vs功率
        valid_indices = df['lmd_totalirrad'].notna() & df['power'].notna()
        if valid_indices.sum() > 1000:
            sample_size = min(5000, valid_indices.sum())
            sample_df = df[valid_indices].sample(sample_size)
            
            axes[0, 1].scatter(sample_df['lmd_totalirrad'], sample_df['power'], alpha=0.3, s=2, color='orange')
            axes[0, 1].set_title('LMD总辐射 vs 功率输出', fontweight='bold')
            axes[0, 1].set_xlabel('总辐射强度 (W/m²)')
            axes[0, 1].set_ylabel('功率输出 (MW)')
            axes[0, 1].grid(True, alpha=0.3)
            
            corr = sample_df['lmd_totalirrad'].corr(sample_df['power'])
            axes[0, 1].text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=axes[0, 1].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 3. 温度vs功率
        valid_indices = df['nwp_temperature'].notna() & df['power'].notna()
        if valid_indices.sum() > 1000:
            sample_size = min(5000, valid_indices.sum())
            sample_df = df[valid_indices].sample(sample_size)
            
            axes[0, 2].scatter(sample_df['nwp_temperature'], sample_df['power'], alpha=0.3, s=2, color='red')
            axes[0, 2].set_title('NWP温度 vs 功率输出', fontweight='bold')
            axes[0, 2].set_xlabel('环境温度 (°C)')
            axes[0, 2].set_ylabel('功率输出 (MW)')
            axes[0, 2].grid(True, alpha=0.3)
            
            corr = sample_df['nwp_temperature'].corr(sample_df['power'])
            axes[0, 2].text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=axes[0, 2].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 4-6. 气象参数分布
        axes[1, 0].hist(df['nwp_globalirrad'].dropna(), bins=40, alpha=0.7, color='yellow', edgecolor='black')
        axes[1, 0].set_title('NWP全球辐射强度分布', fontweight='bold')
        axes[1, 0].set_xlabel('辐射强度 (W/m²)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(df['nwp_temperature'].dropna(), bins=40, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('NWP环境温度分布', fontweight='bold')
        axes[1, 1].set_xlabel('环境温度 (°C)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].hist(df['nwp_humidity'].dropna(), bins=40, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 2].set_title('NWP相对湿度分布', fontweight='bold')
        axes[1, 2].set_xlabel('相对湿度 (%)')
        axes[1, 2].set_ylabel('频次')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 小时功率vs辐射热图
        if 'hour' in df.columns:
            # 创建小时-辐射的功率热图
            df_clean = df[df['nwp_globalirrad'].notna() & df['power'].notna() & df['hour'].notna()]
            if len(df_clean) > 100:
                # 将辐射值分组
                df_clean['irrad_bins'] = pd.cut(df_clean['nwp_globalirrad'], bins=10)
                heatmap_data = df_clean.groupby(['hour', 'irrad_bins'])['power'].mean().reset_index()
                pivot_data = heatmap_data.pivot(index='irrad_bins', columns='hour', values='power')
                
                im = axes[2, 0].imshow(pivot_data.values, aspect='auto', cmap='YlOrRd')
                axes[2, 0].set_title('小时-辐射功率热力图', fontweight='bold')
                axes[2, 0].set_xlabel('小时 (0-23)')
                axes[2, 0].set_ylabel('辐射强度区间')
                
                # 设置坐标轴
                axes[2, 0].set_xticks(range(len(pivot_data.columns)))
                axes[2, 0].set_xticklabels(pivot_data.columns)
                axes[2, 0].set_yticks(range(len(pivot_data.index)))
                axes[2, 0].set_yticklabels([f'{i:.0f}-{j:.0f}' for i,j in 
                                          [(interval.left, interval.right) for interval in pivot_data.index]])
                
                plt.colorbar(im, ax=axes[2, 0], label='平均功率 (MW)')
        
        # 8. 季节性发电vs辐射
        if 'season' in df.columns:
            season_data = df.groupby('season').agg({
                'power': 'mean',
                'nwp_globalirrad': 'mean',
                'nwp_temperature': 'mean'
            }).reset_index()
            
            if len(season_data) > 1:
                seasons = season_data['season']
                x_pos = np.arange(len(seasons))
                
                ax1 = axes[2, 1]
                ax2 = ax1.twinx()
                
                bars1 = ax1.bar(x_pos - 0.2, season_data['power'], 0.4, label='平均功率', alpha=0.7, color='green')
                bars2 = ax2.bar(x_pos + 0.2, season_data['nwp_globalirrad'], 0.4, label='平均辐射', alpha=0.7, color='orange')
                
                ax1.set_title('季节性发电vs辐射对比', fontweight='bold')
                ax1.set_xlabel('季节')
                ax1.set_ylabel('平均功率 (MW)', color='green')
                ax2.set_ylabel('平均辐射 (W/m²)', color='orange')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(seasons)
                
                # 添加图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 9. 相关性矩阵热图
        corr_cols = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                     'nwp_humidity', 'lmd_totalirrad', 'power']
        available_cols = [col for col in corr_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            corr_matrix = df[available_cols].corr()
            
            im = axes[2, 2].imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
            axes[2, 2].set_title('气象-发电相关性矩阵', fontweight='bold')
            
            # 设置标签
            col_labels = [col.replace('nwp_', 'NWP_').replace('lmd_', 'LMD_').replace('_', '\n') for col in available_cols]
            axes[2, 2].set_xticks(range(len(available_cols)))
            axes[2, 2].set_yticks(range(len(available_cols)))
            axes[2, 2].set_xticklabels(col_labels, rotation=45)
            axes[2, 2].set_yticklabels(col_labels)
            
            # 添加数值标注
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                    axes[2, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                   ha='center', va='center', color=text_color, fontweight='bold')
            
            plt.colorbar(im, ax=axes[2, 2], label='相关系数')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pvod_meteorological_correlation_analysis_fixed.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ 气象相关性分析图表已保存")
    
    def generate_summary_report(self, station_data: Dict, metadata: pd.DataFrame):
        """生成汇总报告"""
        print("\n" + "="*60)
        print("PVOD数据集分析汇总报告 (修正版)")
        print("="*60)
        
        # 基本信息
        total_stations = len(station_data)
        total_capacity_kw = metadata['Capacity'].sum()
        total_capacity_mw = total_capacity_kw / 1000
        
        print(f"\n📊 基本信息:")
        print(f"  总站点数: {total_stations}")
        print(f"  总装机容量: {total_capacity_kw:,} kW ({total_capacity_mw:.1f} MW)")
        
        # 技术类型分布
        tech_dist = metadata['PV_Technology'].value_counts()
        print(f"\n🔧 技术类型分布:")
        for tech, count in tech_dist.items():
            capacity_by_tech = metadata[metadata['PV_Technology'] == tech]['Capacity'].sum() / 1000
            print(f"  {tech}: {count} 个站点, 总容量 {capacity_by_tech:.1f} MW")
        
        # 地理分布
        lon_range = metadata['Longitude'].max() - metadata['Longitude'].min()
        lat_range = metadata['Latitude'].max() - metadata['Latitude'].min()
        print(f"\n🌍 地理分布:")
        print(f"  经度范围: {metadata['Longitude'].min():.2f}° - {metadata['Longitude'].max():.2f}° (跨度{lon_range:.2f}°)")
        print(f"  纬度范围: {metadata['Latitude'].min():.2f}° - {metadata['Latitude'].max():.2f}° (跨度{lat_range:.2f}°)")
        print(f"  地理中心: ({metadata['Longitude'].mean():.2f}°, {metadata['Latitude'].mean():.2f}°)")
        
        # 性能统计
        all_cf = []
        all_power = []
        data_summary = []
        
        for station_id, df in station_data.items():
            capacity_mw = df['capacity_mw'].iloc[0]
            avg_power = df['power'].mean()
            max_power = df['power'].max()
            avg_cf = df['capacity_factor'].mean()
            max_cf = df['capacity_factor'].max()
            data_count = len(df)
            
            all_cf.append(avg_cf)
            all_power.append(avg_power)
            
            # 时间范围
            if 'date_time' in df.columns:
                time_span = (df['date_time'].max() - df['date_time'].min()).days
                start_date = df['date_time'].min().strftime('%Y-%m-%d')
                end_date = df['date_time'].max().strftime('%Y-%m-%d')
            else:
                time_span = 0
                start_date = end_date = "未知"
            
            # 发电时间统计（功率>0的时间）
            operating_hours = (df['power'] > 0).sum()
            operating_ratio = (operating_hours / len(df)) * 100
            
            data_summary.append({
                'station': station_id,
                'capacity_mw': capacity_mw,
                'avg_power': avg_power,
                'max_power': max_power,
                'avg_cf': avg_cf,
                'max_cf': max_cf,
                'data_count': data_count,
                'time_span': time_span,
                'start_date': start_date,
                'end_date': end_date,
                'operating_hours': operating_hours,
                'operating_ratio': operating_ratio
            })
        
        print(f"\n⚡ 整体性能:")
        print(f"  平均容量因子: {np.mean(all_cf):.2f}%")
        print(f"  容量因子范围: {np.min(all_cf):.2f}% - {np.max(all_cf):.2f}%")
        print(f"  总平均功率: {np.sum(all_power):.1f} MW")
        print(f"  整体容量因子: {(np.sum(all_power)/total_capacity_mw)*100:.2f}%")
        print(f"  功率标准差: {np.std(all_power):.2f} MW")
        
        # 各站点详细信息
        print(f"\n📋 各站点详细信息:")
        print("-" * 140)
        print(f"{'站点':<12} {'容量(MW)':<10} {'平均功率':<10} {'最大功率':<10} {'容量因子':<10} {'最大CF':<10} {'运行时间':<10} {'数据量':<8} {'时间跨度':<8}")
        print("-" * 140)
        
        # 按容量因子排序
        data_summary.sort(key=lambda x: x['avg_cf'], reverse=True)
        
        for info in data_summary:
            print(f"{info['station']:<12} {info['capacity_mw']:<10.1f} "
                  f"{info['avg_power']:<10.2f} {info['max_power']:<10.2f} "
                  f"{info['avg_cf']:<10.1f}% {info['max_cf']:<10.1f}% "
                  f"{info['operating_ratio']:<10.1f}% {info['data_count']:<8} {info['time_span']:<8}天")
        
        print("-" * 140)
        
        # 性能排名
        best_station = max(data_summary, key=lambda x: x['avg_cf'])
        worst_station = min(data_summary, key=lambda x: x['avg_cf'])
        highest_max = max(data_summary, key=lambda x: x['max_power'])
        
        print(f"\n🏆 性能排名:")
        print(f"  最佳平均性能: {best_station['station']} (容量因子: {best_station['avg_cf']:.1f}%)")
        print(f"  最差平均性能: {worst_station['station']} (容量因子: {worst_station['avg_cf']:.1f}%)")
        print(f"  最高功率输出: {highest_max['station']} (最大功率: {highest_max['max_power']:.1f} MW)")
        
        # 时间分析
        print(f"\n📅 时间分析:")
        all_start_dates = [info['start_date'] for info in data_summary if info['start_date'] != '未知']
        all_end_dates = [info['end_date'] for info in data_summary if info['end_date'] != '未知']
        
        if all_start_dates and all_end_dates:
            earliest_start = min(all_start_dates)
            latest_end = max(all_end_dates)
            print(f"  数据时间范围: {earliest_start} ~ {latest_end}")
            
            avg_operating_ratio = np.mean([info['operating_ratio'] for info in data_summary])
            print(f"  平均运行时间比例: {avg_operating_ratio:.1f}%")
        
        # 数据质量评估
        print(f"\n📈 数据质量:")
        total_records = sum(info['data_count'] for info in data_summary)
        avg_records = total_records / len(data_summary)
        print(f"  总数据记录: {total_records:,}")
        print(f"  平均每站点: {avg_records:.0f} 条记录")
        print(f"  数据时间分辨率: 15分钟间隔")
        
        # 计算各站点功率数据缺失率
        missing_rates = []
        for station_id, df in station_data.items():
            missing_rate = (df['power'].isna().sum() / len(df)) * 100
            missing_rates.append(missing_rate)
            if missing_rate > 5:
                print(f"  ⚠️  {station_id} 功率数据缺失率: {missing_rate:.1f}%")
        
        avg_missing_rate = np.mean(missing_rates)
        print(f"  平均功率数据缺失率: {avg_missing_rate:.3f}%")
        
        # 技术类型性能比较
        print(f"\n🔬 技术类型性能:")
        tech_performance = {}
        for station_id, df in station_data.items():
            tech = df['technology'].iloc[0]
            avg_cf = df['capacity_factor'].mean()
            if tech not in tech_performance:
                tech_performance[tech] = []
            tech_performance[tech].append(avg_cf)
        
        for tech, cf_list in tech_performance.items():
            avg_cf = np.mean(cf_list)
            std_cf = np.std(cf_list)
            print(f"  {tech}: 平均容量因子 {avg_cf:.2f}% (±{std_cf:.2f}%)")
        
        print("\n" + "="*60)
        print("✅ 分析完成！所有图表已保存到 Figures 目录")
        print("📁 生成的文件:")
        print("  📊 pvod_overview_analysis_fixed.png - 数据概览分析")
        print("  ⚡ pvod_power_analysis_fixed.png - 发电分析")
        print("  🌡️ pvod_meteorological_correlation_analysis_fixed.png - 气象相关性分析")
        print("  🧪 chinese_font_test_simplified.png - 中文字体测试")
        print("="*60)
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("🚀 开始PVOD数据集全面分析 (修正版 - 简化中文字体设置)...")
        
        # 测试中文字体
        self.test_chinese_display()
        
        # 加载所有数据
        station_data, metadata = self.load_all_stations()
        
        if not station_data:
            print("❌ 未找到有效数据，分析终止")
            return
        
        print(f"\n✅ 成功加载 {len(station_data)} 个站点的数据")
        
        # 执行各种分析
        self.create_overview_analysis(station_data, metadata)
        self.create_power_generation_analysis(station_data)
        self.create_meteorological_correlation_analysis(station_data)
        
        # 生成汇总报告
        self.generate_summary_report(station_data, metadata)


def main():
    """主函数"""
    try:
        analyzer = PVODAnalyzerCorrectedFixed()
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 