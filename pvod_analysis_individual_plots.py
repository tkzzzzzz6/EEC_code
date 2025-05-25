# PVOD数据集分析器 (独立图片版 - 简化中文字体设置)
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
import os

# 简化的中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')

# 设置警告
warnings.filterwarnings('ignore')

class PVODAnalyzerIndividualPlots:
    """PVOD光伏数据集分析器 (独立图片版 - 简化中文字体设置)"""
    
    def __init__(self, data_dir: str = "PVODdatasets_v1.0", figures_dir: str = "Figures"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.output_dir = self.figures_dir / "pvod_individual_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建中文子目录
        self.overview_dir = self.output_dir / "概览分析"
        self.power_dir = self.output_dir / "发电分析"
        self.meteorological_dir = self.output_dir / "气象分析"
        
        # 创建所有子目录
        self.overview_dir.mkdir(exist_ok=True)
        self.power_dir.mkdir(exist_ok=True) 
        self.meteorological_dir.mkdir(exist_ok=True)
        
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
    
    def ensure_chinese_font(self):
        """确保中文字体设置正确应用"""
        mpl.rc('font', family='simhei')
        plt.rcParams['axes.unicode_minus'] = False
        
    def save_individual_plot(self, fig, filename, title="", subdir=None):
        """保存单个图片并确保字体设置"""
        self.ensure_chinese_font()
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 确定保存路径
        if subdir:
            output_path = subdir / filename
        else:
            output_path = self.output_dir / filename
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ 已保存: {filename}")
    
    def test_chinese_display(self):
        """测试中文显示效果"""
        print("🧪 测试中文字体显示...")
        
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
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
        
        self.save_individual_plot(fig, '中文字体测试图.png', subdir=self.overview_dir)
        print("✓ 中文字体测试图已保存")
        
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
                # 添加容量信息
                capacity_kw = row['Capacity']
                capacity_mw = capacity_kw / 1000
                
                df['capacity_kw'] = capacity_kw
                df['capacity_mw'] = capacity_mw
                df['capacity_factor'] = (df['power'] / capacity_mw) * 100
                df['technology'] = row['PV_Technology']
                df['longitude'] = row['Longitude']
                df['latitude'] = row['Latitude']
                
                station_data[station_id] = df
                print(f"{station_id} 数据加载成功，形状: {df.shape}")
        
        return station_data, metadata
    
    def create_individual_overview_plots(self, station_data: Dict, metadata: pd.DataFrame):
        """创建独立的概览分析图表"""
        print("创建独立概览分析图表...")
        
        # 1. 各站点装机容量
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        capacities = metadata['Capacity'].values / 1000
        station_names = metadata['Station_ID'].values
        
        bars = ax.bar(range(len(station_names)), capacities, alpha=0.8, color='skyblue', edgecolor='navy')
        ax.set_title('各站点装机容量分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('光伏发电站点编号', fontsize=12)
        ax.set_ylabel('装机容量 (MW)', fontsize=12)
        ax.set_xticks(range(len(station_names)))
        ax.set_xticklabels(station_names, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值
        for bar, cap in zip(bars, capacities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{cap:.1f} MW', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        self.save_individual_plot(fig, '01_各站点装机容量分析.png', subdir=self.overview_dir)
        
        # 2. 地理分布
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        lons = metadata['Longitude'].values
        lats = metadata['Latitude'].values
        scatter = ax.scatter(lons, lats, s=capacities*20, alpha=0.7, c=capacities, cmap='viridis', edgecolors='black')
        ax.set_title('PVOD光伏站点地理分布图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('经度 (°)', fontsize=12)
        ax.set_ylabel('纬度 (°)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('装机容量 (MW)', fontsize=12)
        
        # 添加站点标签
        for i, name in enumerate(station_names):
            ax.annotate(name, (lons[i], lats[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        self.save_individual_plot(fig, '02_光伏站点地理分布图.png', subdir=self.overview_dir)
        
        # 3. 技术类型分布
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        tech_counts = metadata['PV_Technology'].value_counts()
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        wedges, texts, autotexts = ax.pie(tech_counts.values, labels=tech_counts.index, 
                                         autopct='%1.1f%%', colors=colors[:len(tech_counts)],
                                         explode=[0.05]*len(tech_counts), shadow=True, startangle=90)
        ax.set_title('光伏技术类型分布', fontsize=16, fontweight='bold', pad=20)
        
        # 美化饼图文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        self.save_individual_plot(fig, '03_光伏技术类型分布.png', subdir=self.overview_dir)
        
        # 4. 数据时间跨度
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        time_spans = []
        for station_id, df in station_data.items():
            if 'date_time' in df.columns:
                time_span = (df['date_time'].max() - df['date_time'].min()).days
                time_spans.append(time_span)
        
        if time_spans:
            bars = ax.bar(range(len(time_spans)), time_spans, alpha=0.8, color='orange', edgecolor='darkorange')
            ax.set_title('各站点数据时间跨度分析', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('光伏发电站点编号', fontsize=12)
            ax.set_ylabel('数据时间跨度 (天)', fontsize=12)
            ax.set_xticks(range(len(station_names)))
            ax.set_xticklabels(station_names, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, span in zip(bars, time_spans):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{span}天', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        self.save_individual_plot(fig, '04_各站点数据时间跨度分析.png', subdir=self.overview_dir)
        
        # 5. 数据量分布
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        data_counts = [len(df) for df in station_data.values()]
        
        bars = ax.bar(range(len(data_counts)), data_counts, alpha=0.8, color='green', edgecolor='darkgreen')
        ax.set_title('各站点数据记录量分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('光伏发电站点编号', fontsize=12)
        ax.set_ylabel('数据记录数量', fontsize=12)
        ax.set_xticks(range(len(station_names)))
        ax.set_xticklabels(station_names, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, data_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                   f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)
        
        self.save_individual_plot(fig, '05_各站点数据记录量分析.png', subdir=self.overview_dir)
        
        # 6. 各站点平均容量因子
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        avg_cf = [station_data[station_id]['capacity_factor'].mean() for station_id in station_names]
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(avg_cf)))
        bars = ax.bar(range(len(station_names)), avg_cf, alpha=0.8, color=colors, edgecolor='black')
        ax.set_title('各站点平均容量因子性能分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('光伏发电站点编号', fontsize=12)
        ax.set_ylabel('平均容量因子 (%)', fontsize=12)
        ax.set_xticks(range(len(station_names)))
        ax.set_xticklabels(station_names, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, cf in zip(bars, avg_cf):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{cf:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        self.save_individual_plot(fig, '06_各站点平均容量因子性能分析.png', subdir=self.overview_dir)
        
        print("✓ 所有概览分析图表已保存")
    
    def create_individual_power_plots(self, station_data: Dict):
        """创建独立的发电分析图表"""
        print("创建独立发电分析图表...")
        
        # 收集所有站点数据
        all_power = []
        all_cf = []
        all_hours = []
        all_months = []
        
        for station_id, df in station_data.items():
            power_data = df['power'].dropna()
            cf_data = df['capacity_factor'].dropna()
            
            all_power.extend(power_data)
            all_cf.extend(cf_data)
            
            if 'hour' in df.columns:
                all_hours.extend(df['hour'])
            if 'month' in df.columns:
                all_months.extend(df['month'])
        
        # 1. 功率分布直方图
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        n, bins, patches = ax.hist(all_power, bins=50, alpha=0.8, color='skyblue', edgecolor='navy')
        ax.set_title('PVOD光伏功率输出分布分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('功率输出 (MW)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_power = np.mean(all_power)
        max_power = np.max(all_power)
        ax.axvline(mean_power, color='red', linestyle='--', linewidth=2,
                  label=f'平均值: {mean_power:.2f} MW')
        ax.axvline(max_power, color='green', linestyle='--', linewidth=2,
                  label=f'最大值: {max_power:.2f} MW')
        ax.legend(fontsize=12)
        
        self.save_individual_plot(fig, '01_光伏功率输出分布分析.png', subdir=self.power_dir)
        
        # 2. 容量因子分布
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        n, bins, patches = ax.hist(all_cf, bins=50, alpha=0.8, color='orange', edgecolor='darkorange')
        ax.set_title('PVOD光伏容量因子分布分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('容量因子 (%)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_cf = np.mean(all_cf)
        ax.axvline(mean_cf, color='red', linestyle='--', linewidth=2,
                  label=f'平均值: {mean_cf:.2f}%')
        ax.legend(fontsize=12)
        
        self.save_individual_plot(fig, '02_光伏容量因子分布分析.png', subdir=self.power_dir)
        
        # 3. 日内发电模式
        if all_hours:
            self.ensure_chinese_font()
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            hour_power = pd.DataFrame({'hour': all_hours, 'power': all_power[:len(all_hours)]})
            hourly_avg = hour_power.groupby('hour')['power'].mean()
            
            ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=3, 
                   markersize=8, color='#FF6B6B', markerfacecolor='white', markeredgewidth=2)
            ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='#FF6B6B')
            ax.set_title('PVOD光伏24小时日内发电模式', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('时间 (小时)', fontsize=12)
            ax.set_ylabel('平均功率输出 (MW)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 23)
            
            # 标注峰值时间
            peak_hour = hourly_avg.idxmax()
            peak_power = hourly_avg.max()
            ax.annotate(f'峰值时间: {peak_hour}:00\n峰值功率: {peak_power:.2f} MW', 
                       xy=(peak_hour, peak_power), xytext=(peak_hour+3, peak_power+1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8),
                       fontsize=12, fontweight='bold')
            
            self.save_individual_plot(fig, '03_光伏24小时日内发电模式.png', subdir=self.power_dir)
        
        # 4. 月度发电模式
        if all_months:
            self.ensure_chinese_font()
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            month_power = pd.DataFrame({'month': all_months, 'power': all_power[:len(all_months)]})
            monthly_avg = month_power.groupby('month')['power'].mean()
            month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                          '7月', '8月', '9月', '10月', '11月', '12月']
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(monthly_avg)))
            bars = ax.bar(monthly_avg.index, monthly_avg.values, alpha=0.8, color=colors, edgecolor='black')
            ax.set_title('PVOD光伏月度发电模式分析', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('月份', fontsize=12)
            ax.set_ylabel('平均功率输出 (MW)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 12.5)
            
            # 设置x轴标签
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([month_names[i-1] for i in range(1, 13)], rotation=45)
            
            # 在柱状图上添加数值
            for bar, month in zip(bars, monthly_avg.index):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            self.save_individual_plot(fig, '04_光伏月度发电模式分析.png', subdir=self.power_dir)
        
        # 5. 各站点功率分布箱线图
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        power_by_station = []
        station_names = list(station_data.keys())
        
        # 添加数据诊断
        print("\n🔍 各站点白天发电数据诊断:")
        
        for station in station_names:
            # 只考虑功率大于0的数据点，排除夜晚不发电的时段
            power_data = station_data[station]['power'].dropna()
            daytime_power = power_data[power_data > 0]  # 过滤掉夜晚功率为0的数据
            
            # 数据诊断
            capacity_mw = station_data[station]['capacity_mw'].iloc[0]
            max_power = daytime_power.max() if len(daytime_power) > 0 else 0
            mean_power = daytime_power.mean() if len(daytime_power) > 0 else 0
            
            print(f"  {station}: 容量={capacity_mw:.1f}MW, 白天数据点={len(daytime_power)}, "
                  f"最大功率={max_power:.2f}MW, 平均功率={mean_power:.2f}MW, "
                  f"最大/容量比={max_power/capacity_mw:.2f}")
            
            # 检查是否有超过容量的异常值
            if max_power > capacity_mw * 1.1:  # 超过容量10%认为异常
                print(f"    ⚠️  {station} 存在超容量发电数据！最大功率{max_power:.2f}MW > 容量{capacity_mw:.1f}MW")
                # 限制最大功率为容量的105%
                daytime_power = daytime_power.clip(upper=capacity_mw * 1.05)
                print(f"    ✅ 已将{station}的功率限制在{capacity_mw * 1.05:.2f}MW以下")
            
            if len(daytime_power) > 0:
                power_by_station.append(daytime_power)
            else:
                # 如果没有有效数据，添加一个很小的值避免空数据
                power_by_station.append([0.001])
        
        box_plot = ax.boxplot(power_by_station, labels=station_names, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('各站点白天发电功率输出分布箱线图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('光伏发电站点编号', fontsize=12)
        ax.set_ylabel('功率输出 (MW)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加说明文字
        ax.text(0.02, 0.98, '注：仅包含白天发电时段数据（功率>0）\n异常超容量数据已被修正', 
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
               fontsize=10, fontweight='bold')
        
        self.save_individual_plot(fig, '05_各站点白天发电功率输出分布箱线图.png', subdir=self.power_dir)
        
        # 6. 各站点发电时间序列对比 - 改进版
        self.ensure_chinese_font()
        
        # 方案1: 月度聚合时间序列
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 收集月度数据
        monthly_data = {}
        for station_id, df in station_data.items():
            if 'date_time' in df.columns and len(df) > 100:
                df_copy = df.copy()
                df_copy['year_month'] = df_copy['date_time'].dt.to_period('M')
                monthly_avg = df_copy.groupby('year_month')['power'].mean()
                if len(monthly_avg) > 0:
                    monthly_data[station_id] = monthly_avg
        
        # 上半部分：月度平均功率趋势
        colors = plt.cm.tab10(np.linspace(0, 1, len(monthly_data)))
        
        for i, (station_id, monthly_series) in enumerate(monthly_data.items()):
            dates = [pd.to_datetime(str(period)) for period in monthly_series.index]
            ax1.plot(dates, monthly_series.values, 
                    marker='o', linewidth=2, markersize=4, 
                    label=station_id, color=colors[i], alpha=0.8)
        
        ax1.set_title('各站点月度平均发电功率趋势对比', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('时间', fontsize=12)
        ax1.set_ylabel('月度平均功率 (MW)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 格式化日期轴
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 下半部分：选择表现最好和最差的几个站点的日度数据样本
        if len(station_data) >= 3:
            # 计算各站点平均功率，选择最好、中等、最差的3个站点
            station_avg_power = {}
            for station_id, df in station_data.items():
                avg_power = df['power'].mean()
                station_avg_power[station_id] = avg_power
            
            sorted_stations = sorted(station_avg_power.items(), key=lambda x: x[1], reverse=True)
            selected_stations = [sorted_stations[0][0], sorted_stations[len(sorted_stations)//2][0], sorted_stations[-1][0]]
            
            ax2.set_title(f'代表性站点详细功率时间序列 (最佳/中等/最差)', fontsize=16, fontweight='bold', pad=20)
            
            selected_colors = ['green', 'orange', 'red']
            for i, station_id in enumerate(selected_stations):
                df = station_data[station_id]
                if 'date_time' in df.columns and len(df) > 100:
                    # 取更多样本但仍然控制密度
                    sample_size = min(2000, len(df))
                    sample_df = df.sample(sample_size).sort_values('date_time')
                    
                    ax2.plot(sample_df['date_time'], sample_df['power'], 
                           alpha=0.7, linewidth=1, label=f'{station_id} (平均{station_avg_power[station_id]:.2f}MW)', 
                           color=selected_colors[i])
            
            ax2.set_xlabel('时间', fontsize=12)
            ax2.set_ylabel('功率输出 (MW)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # 格式化日期轴
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        self.save_individual_plot(fig, '06_各站点发电功率时间序列对比_改进版.png', subdir=self.power_dir)
        
        # 方案2: 热力图展示所有站点的月度表现
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # 创建月度功率矩阵
        all_months = set()
        for monthly_series in monthly_data.values():
            all_months.update(monthly_series.index)
        all_months = sorted(list(all_months))
        
        # 构建矩阵数据
        matrix_data = []
        station_labels = []
        
        for station_id in sorted(station_data.keys()):
            if station_id in monthly_data:
                row_data = []
                for month in all_months:
                    if month in monthly_data[station_id]:
                        row_data.append(monthly_data[station_id][month])
                    else:
                        row_data.append(0)  # 没有数据的月份填0
                matrix_data.append(row_data)
                station_labels.append(station_id)
        
        if matrix_data:
            matrix_data = np.array(matrix_data)
            
            # 创建热力图
            im = ax.imshow(matrix_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            
            ax.set_title('各站点月度平均发电功率热力图', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('月份', fontsize=12)
            ax.set_ylabel('光伏发电站点', fontsize=12)
            
            # 设置坐标轴
            ax.set_yticks(range(len(station_labels)))
            ax.set_yticklabels(station_labels)
            ax.set_xticks(range(len(all_months)))
            ax.set_xticklabels([str(month) for month in all_months], rotation=45)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('月度平均功率 (MW)', fontsize=12)
            
            # 在热力图上添加数值
            for i in range(len(station_labels)):
                for j in range(len(all_months)):
                    if matrix_data[i, j] > 0:
                        text_color = 'white' if matrix_data[i, j] > matrix_data.max() * 0.5 else 'black'
                        ax.text(j, i, f'{matrix_data[i, j]:.1f}', 
                               ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
        
        self.save_individual_plot(fig, '07_各站点月度发电功率热力图.png', subdir=self.power_dir)
        
        print("✓ 所有发电分析图表已保存")
    
    def create_individual_meteorological_plots(self, station_data: Dict):
        """创建独立的气象相关性分析图表"""
        print("创建独立气象分析图表...")
        
        # 选择数据量最大的站点进行分析
        max_data_station = max(station_data.items(), key=lambda x: len(x[1]))
        station_id, df = max_data_station
        
        print(f"使用 {station_id} 站点数据进行气象分析（数据量最大: {len(df)} 条记录）")
        
        # 1. NWP全球辐射vs功率散点图
        valid_indices = df['nwp_globalirrad'].notna() & df['power'].notna()
        if valid_indices.sum() > 1000:
            self.ensure_chinese_font()
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            sample_size = min(5000, valid_indices.sum())
            sample_df = df[valid_indices].sample(sample_size)
            
            scatter = ax.scatter(sample_df['nwp_globalirrad'], sample_df['power'], 
                               alpha=0.5, s=15, c=sample_df['power'], cmap='viridis')
            ax.set_title(f'NWP全球辐射与功率输出相关性分析 ({station_id})', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('NWP全球辐射强度 (W/m²)', fontsize=12)
            ax.set_ylabel('功率输出 (MW)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 计算相关系数
            corr = sample_df['nwp_globalirrad'].corr(sample_df['power'])
            ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                   fontsize=14, fontweight='bold')
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('功率输出 (MW)', fontsize=12)
            
            self.save_individual_plot(fig, '01_NWP全球辐射与功率输出相关性分析.png', subdir=self.meteorological_dir)
        
        # 2. LMD总辐射vs功率散点图
        valid_indices = df['lmd_totalirrad'].notna() & df['power'].notna()
        if valid_indices.sum() > 1000:
            self.ensure_chinese_font()
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            sample_size = min(5000, valid_indices.sum())
            sample_df = df[valid_indices].sample(sample_size)
            
            scatter = ax.scatter(sample_df['lmd_totalirrad'], sample_df['power'], 
                               alpha=0.5, s=15, c=sample_df['power'], cmap='plasma')
            ax.set_title(f'LMD总辐射与功率输出相关性分析 ({station_id})', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('LMD总辐射强度 (W/m^2)', fontsize=12)
            ax.set_ylabel('功率输出 (MW)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            corr = sample_df['lmd_totalirrad'].corr(sample_df['power'])
            ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
                   fontsize=14, fontweight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('功率输出 (MW)', fontsize=12)
            
            self.save_individual_plot(fig, '02_LMD总辐射与功率输出相关性分析.png', subdir=self.meteorological_dir)
        
        # 3. 温度vs功率散点图
        valid_indices = df['nwp_temperature'].notna() & df['power'].notna()
        if valid_indices.sum() > 1000:
            self.ensure_chinese_font()
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            sample_size = min(5000, valid_indices.sum())
            sample_df = df[valid_indices].sample(sample_size)
            
            scatter = ax.scatter(sample_df['nwp_temperature'], sample_df['power'], 
                               alpha=0.5, s=15, c=sample_df['power'], cmap='coolwarm')
            ax.set_title(f'环境温度与功率输出相关性分析 ({station_id})', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('NWP环境温度 (°C)', fontsize=12)
            ax.set_ylabel('功率输出 (MW)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            corr = sample_df['nwp_temperature'].corr(sample_df['power'])
            ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8),
                   fontsize=14, fontweight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('功率输出 (MW)', fontsize=12)
            
            self.save_individual_plot(fig, '03_环境温度与功率输出相关性分析.png', subdir=self.meteorological_dir)
        
        # 4. NWP全球辐射分布
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        radiation_data = df['nwp_globalirrad'].dropna()
        n, bins, patches = ax.hist(radiation_data, bins=60, alpha=0.8, color='yellow', edgecolor='orange')
        ax.set_title(f'NWP全球辐射强度分布 ({station_id})', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('NWP全球辐射强度 (W/m²)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_rad = radiation_data.mean()
        max_rad = radiation_data.max()
        ax.axvline(mean_rad, color='red', linestyle='--', linewidth=2,
                  label=f'平均值: {mean_rad:.1f} W/m²')
        ax.axvline(max_rad, color='green', linestyle='--', linewidth=2,
                  label=f'最大值: {max_rad:.1f} W/m²')
        ax.legend(fontsize=12)
        
        self.save_individual_plot(fig, '04_NWP全球辐射强度分布.png', subdir=self.meteorological_dir)
        
        # 5. 环境温度分布
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        temp_data = df['nwp_temperature'].dropna()
        n, bins, patches = ax.hist(temp_data, bins=50, alpha=0.8, color='red', edgecolor='darkred')
        ax.set_title(f'NWP环境温度分布 ({station_id})', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('NWP环境温度 (°C)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_temp = temp_data.mean()
        min_temp = temp_data.min()
        max_temp = temp_data.max()
        ax.axvline(mean_temp, color='blue', linestyle='--', linewidth=2,
                  label=f'平均值: {mean_temp:.1f}°C')
        ax.text(0.95, 0.95, f'温度范围: {min_temp:.1f}°C ~ {max_temp:.1f}°C', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
               fontsize=12, fontweight='bold')
        ax.legend(fontsize=12)
        
        self.save_individual_plot(fig, '05_NWP环境温度分布.png', subdir=self.meteorological_dir)
        
        # 6. 相对湿度分布
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        humidity_data = df['nwp_humidity'].dropna()
        n, bins, patches = ax.hist(humidity_data, bins=50, alpha=0.8, color='blue', edgecolor='darkblue')
        ax.set_title(f'NWP相对湿度分布 ({station_id})', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('NWP相对湿度 (%)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_hum = humidity_data.mean()
        ax.axvline(mean_hum, color='green', linestyle='--', linewidth=2,
                  label=f'平均值: {mean_hum:.1f}%')
        ax.legend(fontsize=12)
        
        self.save_individual_plot(fig, '06_NWP相对湿度分布.png', subdir=self.meteorological_dir)
        
        # 7. 季节性发电vs辐射对比
        if 'season' in df.columns:
            self.ensure_chinese_font()
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            season_data = df.groupby('season').agg({
                'power': 'mean',
                'nwp_globalirrad': 'mean',
                'nwp_temperature': 'mean'
            }).reset_index()
            
            if len(season_data) > 1:
                seasons = season_data['season']
                x_pos = np.arange(len(seasons))
                
                ax2 = ax.twinx()
                
                bars1 = ax.bar(x_pos - 0.2, season_data['power'], 0.4, 
                              label='平均功率', alpha=0.8, color='green', edgecolor='darkgreen')
                bars2 = ax2.bar(x_pos + 0.2, season_data['nwp_globalirrad'], 0.4, 
                               label='平均辐射', alpha=0.8, color='orange', edgecolor='darkorange')
                
                ax.set_title(f'季节性发电与辐射对比分析 ({station_id})', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('季节', fontsize=12)
                ax.set_ylabel('平均功率输出 (MW)', fontsize=12, color='green')
                ax2.set_ylabel('平均辐射强度 (W/m²)', fontsize=12, color='orange')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(seasons)
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, val in zip(bars1, season_data['power']):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                for bar, val in zip(bars2, season_data['nwp_globalirrad']):
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                # 添加图例
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
                
                self.save_individual_plot(fig, '07_季节性发电与辐射对比分析.png', subdir=self.meteorological_dir)
        
        # 8. 气象-发电相关性矩阵热图
        self.ensure_chinese_font()
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        corr_cols = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                     'nwp_humidity', 'nwp_windspeed', 'lmd_totalirrad', 'lmd_diffuseirrad', 
                     'lmd_temperature', 'power']
        available_cols = [col for col in corr_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            corr_matrix = df[available_cols].corr()
            
            # 创建热图
            im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_title(f'气象变量与发电功率相关性矩阵 ({station_id})', fontsize=16, fontweight='bold', pad=20)
            
            # 设置坐标轴标签
            col_labels = []
            for col in available_cols:
                if col == 'power':
                    col_labels.append('功率输出')
                elif 'nwp_globalirrad' in col:
                    col_labels.append('NWP全球\n辐射')
                elif 'nwp_directirrad' in col:
                    col_labels.append('NWP直射\n辐射')
                elif 'nwp_temperature' in col:
                    col_labels.append('NWP\n温度')
                elif 'nwp_humidity' in col:
                    col_labels.append('NWP\n湿度')
                elif 'nwp_windspeed' in col:
                    col_labels.append('NWP\n风速')
                elif 'lmd_totalirrad' in col:
                    col_labels.append('LMD总\n辐射')
                elif 'lmd_diffuseirrad' in col:
                    col_labels.append('LMD散射\n辐射')
                elif 'lmd_temperature' in col:
                    col_labels.append('LMD\n温度')
                else:
                    col_labels.append(col.replace('_', '\n'))
            
            ax.set_xticks(range(len(available_cols)))
            ax.set_yticks(range(len(available_cols)))
            ax.set_xticklabels(col_labels, fontsize=10)
            ax.set_yticklabels(col_labels, fontsize=10)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # 添加数值标注
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('相关系数', fontsize=12)
            
            self.save_individual_plot(fig, '08_气象变量与发电功率相关性矩阵.png', subdir=self.meteorological_dir)
        
        # 9. 小时功率vs辐射热力图
        if 'hour' in df.columns:
            self.ensure_chinese_font()
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # 创建小时-辐射的功率热图
            df_clean = df[df['nwp_globalirrad'].notna() & df['power'].notna() & df['hour'].notna()]
            if len(df_clean) > 100:
                # 将辐射值分组
                df_clean = df_clean.copy()
                df_clean['irrad_bins'] = pd.cut(df_clean['nwp_globalirrad'], bins=10, labels=False)
                
                # 创建透视表
                pivot_data = df_clean.groupby(['hour', 'irrad_bins'])['power'].mean().reset_index()
                pivot_table = pivot_data.pivot(index='irrad_bins', columns='hour', values='power')
                
                # 填充缺失值
                pivot_table = pivot_table.fillna(0)
                
                im = ax.imshow(pivot_table.values, aspect='auto', cmap='YlOrRd', origin='lower')
                ax.set_title(f'小时-辐射强度功率输出热力图 ({station_id})', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('小时 (0-23)', fontsize=12)
                ax.set_ylabel('辐射强度区间', fontsize=12)
                
                # 设置坐标轴
                ax.set_xticks(range(len(pivot_table.columns)))
                ax.set_xticklabels(pivot_table.columns)
                ax.set_yticks(range(len(pivot_table.index)))
                
                # 创建辐射区间标签
                irrad_range = df_clean['nwp_globalirrad'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                irrad_labels = [f'{irrad_range.iloc[i]:.0f}-{irrad_range.iloc[i+1]:.0f}' 
                               for i in range(len(irrad_range)-1)]
                ax.set_yticklabels(irrad_labels[:len(pivot_table.index)], fontsize=8)
                
                plt.colorbar(im, ax=ax, label='平均功率输出 (MW)')
                
                self.save_individual_plot(fig, '09_小时辐射强度功率输出热力图.png', subdir=self.meteorological_dir)
        
        print("✓ 所有气象分析图表已保存")

    def run_individual_analysis(self):
        """运行独立图表分析"""
        print("🚀 开始PVOD数据集独立图表分析...")
        
        # 测试中文字体
        self.test_chinese_display()
        
        # 加载所有数据
        station_data, metadata = self.load_all_stations()
        
        if not station_data:
            print("❌ 未找到有效数据，分析终止")
            return
        
        print(f"\n✅ 成功加载 {len(station_data)} 个站点的数据")
        
        # 执行各种分析
        self.create_individual_overview_plots(station_data, metadata)
        self.create_individual_power_plots(station_data)
        self.create_individual_meteorological_plots(station_data)
        
        print("\n" + "="*60)
        print("✅ 独立图表分析完成！")
        print(f"📁 所有图表已保存到目录: {self.output_dir}")
        
        # 列出所有生成的文件按目录分类
        print(f"\n📊 生成的图表文件按目录分类:")
        
        # 概览分析文件
        overview_files = sorted([f for f in os.listdir(self.overview_dir) if f.endswith('.png')])
        print(f"\n🔧 概览分析 ({len(overview_files)} 个文件):")
        for i, filename in enumerate(overview_files, 1):
            print(f"  {i:2d}. {filename}")
        
        # 发电分析文件
        power_files = sorted([f for f in os.listdir(self.power_dir) if f.endswith('.png')])
        print(f"\n⚡ 发电分析 ({len(power_files)} 个文件):")
        for i, filename in enumerate(power_files, 1):
            print(f"  {i:2d}. {filename}")
        
        # 气象分析文件
        meteorological_files = sorted([f for f in os.listdir(self.meteorological_dir) if f.endswith('.png')])
        print(f"\n🌡️ 气象分析 ({len(meteorological_files)} 个文件):")
        for i, filename in enumerate(meteorological_files, 1):
            print(f"  {i:2d}. {filename}")
        
        total_files = len(overview_files) + len(power_files) + len(meteorological_files)
        print(f"\n📈 总计生成 {total_files} 个独立图表文件")
        
        print("\n📂 目录结构:")
        print(f"  {self.output_dir}/")
        print(f"  ├── 概览分析/ ({len(overview_files)} 个文件)")
        print(f"  ├── 发电分析/ ({len(power_files)} 个文件)")
        print(f"  └── 气象分析/ ({len(meteorological_files)} 个文件)")
        
        print("\n🎨 新增图表说明:")
        print("  📊 发电分析新增:")
        print("    - 05_各站点白天发电功率输出分布箱线图.png")
        print("    - 06_各站点发电功率时间序列对比_改进版.png")
        print("    - 07_各站点月度发电功率热力图.png")
        print("  🌡️ 气象分析新增:")
        print("    - 08_气象变量与发电功率相关性矩阵.png")
        print("    - 09_小时辐射强度功率输出热力图.png")
        
        print("="*60)


def main():
    """主函数"""
    try:
        analyzer = PVODAnalyzerIndividualPlots()
        analyzer.run_individual_analysis()
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 