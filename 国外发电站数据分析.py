# 导入必要的库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np
import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# 设置警告和图表样式
warnings.filterwarnings('ignore', category=FutureWarning)
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
sns.set_palette("husl")

class EnhancedRenewableEnergyAnalyzer:
    """增强版可再生能源数据分析器"""
    
    def __init__(self, data_dir: str = "data_processed", figures_dir: str = "Figures"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # 风电场配置
        self.wind_farm_config = {
            1: {"file": "Wind farm site 1 (Nominal capacity-99MW).xlsx", 
                "capacity": 99,
                "columns": ['time', 'WS_10', 'WD_10', 'WS_30', 'WD_30', 'WS_50', 'WD_50', 
                          'WS_cen', 'WD_cen', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']},
            2: {"file": "Wind farm site 2 (Nominal capacity-200MW).xlsx", 
                "capacity": 200,
                "columns": ['time', 'WS_10', 'WD_10', 'WS_30', 'WD_30', 'WS_50', 'WD_50', 
                          'WS_cen', 'WD_cen', 'Air_T', 'Air_P', 'Power(MW)']},
            3: {"file": "Wind farm site 3 (Nominal capacity-99MW).xlsx", 
                "capacity": 99,
                "columns": ['time', 'WS_10', 'WD_10', 'WS_30', 'WD_30', 'WS_50', 'WD_50', 
                          'WS_cen', 'WD_cen', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']},
            4: {"file": "Wind farm site 4 (Nominal capacity-66MW).xlsx", 
                "capacity": 66,
                "columns": ['time', 'WS_10', 'WD_10', 'WS_30', 'WD_30', 'WS_50', 'WD_50', 
                          'WS_cen', 'WD_cen', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']},
            5: {"file": "Wind farm site 5 (Nominal capacity-36MW).xlsx", 
                "capacity": 36,
                "columns": ['time', 'WS_10', 'WD_10', 'WS_30', 'WD_30', 'WS_50', 'WD_50', 
                          'WS_cen', 'WD_cen', 'Air_T', 'Power(MW)']},
            6: {"file": "Wind farm site 6 (Nominal capacity-96MW).xlsx", 
                "capacity": 96,
                "columns": ['time', 'WS_10', 'WD_10', 'WS_30', 'WD_30', 'WS_50', 'WD_50', 
                          'WS_cen', 'WD_cen', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']}
        }
        
        # 太阳能电站配置
        self.solar_station_config = {
            1: {"file": "Solar station site 1 (Nominal capacity-50MW).xlsx", 
                "capacity": 50,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_T', 'Air_P', 'Power(MW)']},
            2: {"file": "Solar station site 2 (Nominal capacity-130MW).xlsx", 
                "capacity": 130,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_T', 'Air_P', 'Power(MW)']},
            3: {"file": "Solar station site 3 (Nominal capacity-30MW).xlsx", 
                "capacity": 30,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_P', 'Air_H', 'Power(MW)']},
            4: {"file": "Solar station site 4 (Nominal capacity-130MW).xlsx", 
                "capacity": 130,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']},
            5: {"file": "Solar station site 5 (Nominal capacity-110MW).xlsx", 
                "capacity": 110,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']},
            6: {"file": "Solar station site 6 (Nominal capacity-35MW).xlsx", 
                "capacity": 35,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']},
            7: {"file": "Solar station site 7 (Nominal capacity-30MW).xlsx", 
                "capacity": 30,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']},
            8: {"file": "Solar station site 8 (Nominal capacity-30MW).xlsx", 
                "capacity": 30,
                "columns": ['time', 'TSI', 'DNI', 'GHI', 'Air_T', 'Air_P', 'Air_H', 'Power(MW)']}
        }
    
    def load_and_process_data(self, config: Dict, subdirectory: str) -> Dict[int, pd.DataFrame]:
        """加载和处理数据"""
        datasets = {}
        
        for site_id, site_config in config.items():
            try:
                file_path = self.data_dir / subdirectory / site_config["file"]
                print(f"正在加载 {file_path}...")
                
                # 读取数据
                df = pd.read_excel(file_path).drop(index=0)
                df.columns = site_config["columns"]
                
                # 数据清理
                df = self._clean_data(df)
                
                # 时间处理
                df = self._process_time_data(df)
                
                # 添加容量信息
                df['Capacity(MW)'] = site_config["capacity"]
                df['Capacity_Factor(%)'] = (df['Power(MW)'] / site_config["capacity"]) * 100
                
                datasets[site_id] = df
                print(f"站点 {site_id} 数据加载成功，形状: {df.shape}")
                
            except Exception as e:
                print(f"加载站点 {site_id} 数据失败: {e}")
                continue
        
        return datasets
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        # 替换特殊字符
        df = df.replace(['--', '<NULL>', 'NaN'], np.nan)
        
        # 转换数值列为float类型
        numeric_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = [col for col in numeric_columns if col != 'time']
        
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                continue
        
        return df
    
    def _process_time_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理时间数据"""
        try:
            df['time'] = pd.to_datetime(df['time'])
            df['hour'] = df['time'].dt.hour
            df['day'] = df['time'].dt.day
            df['month'] = df['time'].dt.month
            df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                          3: 'Spring', 4: 'Spring', 5: 'Spring',
                                          6: 'Summer', 7: 'Summer', 8: 'Summer',
                                          9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
            df['weekday'] = df['time'].dt.day_name()
        except:
            print("时间数据处理失败")
        
        return df
    
    def create_time_series_analysis(self, datasets: Dict[int, pd.DataFrame], station_type: str):
        """创建时间序列分析"""
        print(f"创建 {station_type} 时间序列分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'{station_type} Time Series Analysis', fontsize=16, fontweight='bold')
        
        for site_id, df in list(datasets.items())[:4]:  # 只分析前4个站点
            if 'time' not in df.columns:
                continue
                
            # 选择一个月的数据进行详细分析
            sample_data = df[df['time'].dt.month == 1].head(1000)
            
            row = (site_id - 1) // 2
            col = (site_id - 1) % 2
            
            axes[row, col].plot(sample_data['time'], sample_data['Power(MW)'], 
                               linewidth=1, alpha=0.8)
            axes[row, col].set_title(f'{station_type} Site {site_id} - Power Output (January)')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel('Power (MW)')
            axes[row, col].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{station_type}_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_power_analysis(self, datasets: Dict[int, pd.DataFrame], station_type: str):
        """创建功率分析"""
        print(f"创建 {station_type} 功率分析...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{station_type} Power Generation Analysis', fontsize=16, fontweight='bold')
        
        # 收集所有站点的功率数据
        all_power_data = []
        all_cf_data = []
        site_labels = []
        
        for site_id, df in datasets.items():
            power_data = df['Power(MW)'].dropna()
            cf_data = df['Capacity_Factor(%)'].dropna()
            all_power_data.extend(power_data)
            all_cf_data.extend(cf_data)
            site_labels.extend([f'Site {site_id}'] * len(power_data))
        
        # 1. 功率分布直方图
        axes[0, 0].hist(all_power_data, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Power Output Distribution')
        axes[0, 0].set_xlabel('Power (MW)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 容量因子分布
        axes[0, 1].hist(all_cf_data, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Capacity Factor Distribution')
        axes[0, 1].set_xlabel('Capacity Factor (%)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. 各站点功率箱线图
        power_by_site = [datasets[site]['Power(MW)'].dropna() for site in datasets.keys()]
        axes[0, 2].boxplot(power_by_site, labels=[f'Site {i}' for i in datasets.keys()])
        axes[0, 2].set_title('Power Output by Site')
        axes[0, 2].set_ylabel('Power (MW)')
        
        # 4. 小时功率变化模式
        hourly_power = {}
        for site_id, df in datasets.items():
            if 'hour' in df.columns:
                hourly_avg = df.groupby('hour')['Power(MW)'].mean()
                axes[1, 0].plot(hourly_avg.index, hourly_avg.values, 
                               label=f'Site {site_id}', marker='o', markersize=3)
        axes[1, 0].set_title('Daily Power Generation Pattern')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Power (MW)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 季节性功率变化
        try:
            # 检查是否有season列
            has_season = any('season' in df.columns for df in datasets.values())
            if has_season:
                seasonal_data = []
                seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
                for season in seasons:
                    season_power = []
                    for df in datasets.values():
                        if 'season' in df.columns:
                            season_data_df = df[df['season'] == season]['Power(MW)'].dropna()
                            season_power.extend(season_data_df)
                    if season_power:  # 只有在有数据时才添加
                        seasonal_data.append(season_power)
                
                if seasonal_data:
                    # 只绘制有数据的季节
                    valid_seasons = seasons[:len(seasonal_data)]
                    axes[1, 1].boxplot(seasonal_data, labels=valid_seasons)
                    axes[1, 1].set_title('Seasonal Power Generation')
                    axes[1, 1].set_ylabel('Power (MW)')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No seasonal data available', 
                                   transform=axes[1, 1].transAxes, ha='center', va='center')
                    axes[1, 1].set_title('Seasonal Power Generation')
            else:
                axes[1, 1].text(0.5, 0.5, 'No seasonal data available', 
                               transform=axes[1, 1].transAxes, ha='center', va='center')
                axes[1, 1].set_title('Seasonal Power Generation')
        except Exception as e:
            print(f"季节性分析失败: {e}")
            axes[1, 1].text(0.5, 0.5, 'Seasonal analysis failed', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Seasonal Power Generation')
        
        # 6. 功率效率分析
        efficiency_data = []
        site_names = []
        for site_id, df in datasets.items():
            avg_power = df['Power(MW)'].mean()
            capacity = df['Capacity(MW)'].iloc[0]
            efficiency = (avg_power / capacity) * 100
            efficiency_data.append(efficiency)
            site_names.append(f'Site {site_id}')
        
        axes[1, 2].bar(site_names, efficiency_data, alpha=0.7, color='green')
        axes[1, 2].set_title('Average Capacity Factor by Site')
        axes[1, 2].set_ylabel('Capacity Factor (%)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{station_type}_power_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_meteorological_analysis(self, datasets: Dict[int, pd.DataFrame], station_type: str):
        """创建气象条件分析"""
        print(f"创建 {station_type} 气象条件分析...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{station_type} Meteorological Analysis', fontsize=16, fontweight='bold')
        
        # 选择第一个站点进行详细分析
        sample_df = list(datasets.values())[0]
        
        if station_type == 'Wind_Farm':
            # 风速分析
            wind_cols = [col for col in sample_df.columns if 'WS_' in col]
            if wind_cols:
                for col in wind_cols[:3]:  # 分析前3个风速列
                    axes[0, 0].hist(sample_df[col].dropna(), alpha=0.5, label=col, bins=30)
                axes[0, 0].set_title('Wind Speed Distribution')
                axes[0, 0].set_xlabel('Wind Speed (m/s)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].legend()
            
            # 风向分析
            wind_dir_cols = [col for col in sample_df.columns if 'WD_' in col]
            if wind_dir_cols:
                wind_dir = sample_df[wind_dir_cols[0]].dropna()
                axes[0, 1].hist(wind_dir, bins=36, alpha=0.7, color='skyblue')
                axes[0, 1].set_title('Wind Direction Distribution')
                axes[0, 1].set_xlabel('Wind Direction (degrees)')
                axes[0, 1].set_ylabel('Frequency')
            
            # 风速vs功率散点图
            if wind_cols and 'Power(MW)' in sample_df.columns:
                wind_speed = sample_df[wind_cols[0]].dropna()
                power = sample_df['Power(MW)'].dropna()
                min_len = min(len(wind_speed), len(power))
                axes[0, 2].scatter(wind_speed[:min_len], power[:min_len], alpha=0.1, s=1)
                axes[0, 2].set_title('Wind Speed vs Power Output')
                axes[0, 2].set_xlabel('Wind Speed (m/s)')
                axes[0, 2].set_ylabel('Power (MW)')
                
        elif station_type == 'Solar_Station':
            # 辐射分析
            irradiance_cols = [col for col in sample_df.columns if any(x in col for x in ['TSI', 'DNI', 'GHI'])]
            if irradiance_cols:
                for col in irradiance_cols:
                    axes[0, 0].hist(sample_df[col].dropna(), alpha=0.5, label=col, bins=30)
                axes[0, 0].set_title('Solar Irradiance Distribution')
                axes[0, 0].set_xlabel('Irradiance (W/m²)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].legend()
            
            # GHI vs 功率散点图
            if 'GHI' in sample_df.columns and 'Power(MW)' in sample_df.columns:
                ghi = sample_df['GHI'].dropna()
                power = sample_df['Power(MW)'].dropna()
                min_len = min(len(ghi), len(power))
                axes[0, 2].scatter(ghi[:min_len], power[:min_len], alpha=0.1, s=1, color='orange')
                axes[0, 2].set_title('Solar Irradiance vs Power Output')
                axes[0, 2].set_xlabel('GHI (W/m²)')
                axes[0, 2].set_ylabel('Power (MW)')
        
        # 温度分析
        if 'Air_T' in sample_df.columns:
            axes[1, 0].hist(sample_df['Air_T'].dropna(), bins=30, alpha=0.7, color='red')
            axes[1, 0].set_title('Air Temperature Distribution')
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 气压分析
        if 'Air_P' in sample_df.columns:
            axes[1, 1].hist(sample_df['Air_P'].dropna(), bins=30, alpha=0.7, color='blue')
            axes[1, 1].set_title('Atmospheric Pressure Distribution')
            axes[1, 1].set_xlabel('Pressure (hPa)')
            axes[1, 1].set_ylabel('Frequency')
        
        # 湿度分析
        if 'Air_H' in sample_df.columns:
            axes[1, 2].hist(sample_df['Air_H'].dropna(), bins=30, alpha=0.7, color='green')
            axes[1, 2].set_title('Relative Humidity Distribution')
            axes[1, 2].set_xlabel('Humidity (%)')
            axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{station_type}_meteorological_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparative_analysis(self, wind_datasets: Dict, solar_datasets: Dict):
        """创建比较分析"""
        print("创建风电vs太阳能比较分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Wind Farm vs Solar Station Comparative Analysis', fontsize=16, fontweight='bold')
        
        # 收集数据
        wind_capacity_factors = []
        solar_capacity_factors = []
        wind_powers = []
        solar_powers = []
        
        for df in wind_datasets.values():
            wind_capacity_factors.extend(df['Capacity_Factor(%)'].dropna())
            wind_powers.extend(df['Power(MW)'].dropna())
        
        for df in solar_datasets.values():
            solar_capacity_factors.extend(df['Capacity_Factor(%)'].dropna())
            solar_powers.extend(df['Power(MW)'].dropna())
        
        # 1. 容量因子比较
        axes[0, 0].hist(wind_capacity_factors, alpha=0.5, label='Wind Farm', bins=50, color='blue')
        axes[0, 0].hist(solar_capacity_factors, alpha=0.5, label='Solar Station', bins=50, color='orange')
        axes[0, 0].set_title('Capacity Factor Comparison')
        axes[0, 0].set_xlabel('Capacity Factor (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. 功率输出比较
        axes[0, 1].boxplot([wind_powers, solar_powers], labels=['Wind Farm', 'Solar Station'])
        axes[0, 1].set_title('Power Output Comparison')
        axes[0, 1].set_ylabel('Power (MW)')
        
        # 3. 日内发电模式比较
        try:
            wind_hourly = []
            solar_hourly = []
            
            for df in wind_datasets.values():
                if 'hour' in df.columns:
                    hourly_avg = df.groupby('hour')['Capacity_Factor(%)'].mean()
                    wind_hourly.append(hourly_avg)
            
            for df in solar_datasets.values():
                if 'hour' in df.columns:
                    hourly_avg = df.groupby('hour')['Capacity_Factor(%)'].mean()
                    solar_hourly.append(hourly_avg)
            
            if wind_hourly and solar_hourly:
                wind_avg = pd.concat(wind_hourly, axis=1).mean(axis=1)
                solar_avg = pd.concat(solar_hourly, axis=1).mean(axis=1)
                
                axes[1, 0].plot(wind_avg.index, wind_avg.values, label='Wind Farm', marker='o', linewidth=2)
                axes[1, 0].plot(solar_avg.index, solar_avg.values, label='Solar Station', marker='s', linewidth=2)
                axes[1, 0].set_title('Daily Generation Pattern Comparison')
                axes[1, 0].set_xlabel('Hour of Day')
                axes[1, 0].set_ylabel('Average Capacity Factor (%)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No hourly data available', 
                               transform=axes[1, 0].transAxes, ha='center', va='center')
                axes[1, 0].set_title('Daily Generation Pattern Comparison')
        except Exception as e:
            print(f"日内模式比较失败: {e}")
            axes[1, 0].text(0.5, 0.5, 'Hourly comparison failed', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Daily Generation Pattern Comparison')
        
        # 4. 效率统计
        if wind_capacity_factors and solar_capacity_factors:
            wind_stats = [np.mean(wind_capacity_factors), np.std(wind_capacity_factors), 
                          np.max(wind_capacity_factors)]
            solar_stats = [np.mean(solar_capacity_factors), np.std(solar_capacity_factors), 
                           np.max(solar_capacity_factors)]
            
            x = ['Mean CF (%)', 'Std CF (%)', 'Max CF (%)']
            x_pos = np.arange(len(x))
            
            axes[1, 1].bar(x_pos - 0.2, wind_stats, 0.4, label='Wind Farm', alpha=0.7)
            axes[1, 1].bar(x_pos + 0.2, solar_stats, 0.4, label='Solar Station', alpha=0.7)
            axes[1, 1].set_title('Performance Statistics Comparison')
            axes[1, 1].set_ylabel('Capacity Factor (%)')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(x)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No capacity factor data available', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Performance Statistics Comparison')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_heatmaps(self, datasets: Dict[int, pd.DataFrame], 
                                  station_type: str, grid_shape: Tuple[int, int]):
        """创建相关性热图"""
        try:
            fig, axes = plt.subplots(*grid_shape, figsize=(20, 20))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            
            # 确保axes是2D数组
            if len(grid_shape) == 1 or grid_shape[0] == 1:
                axes = axes.reshape(1, -1)
            elif grid_shape[1] == 1:
                axes = axes.reshape(-1, 1)
            
            for i, (site_id, data) in enumerate(datasets.items()):
                row = i // grid_shape[1]
                col = i % grid_shape[1]
                
                # 只选择数值列进行相关性分析
                numeric_data = data.select_dtypes(include=[np.number])
                
                if numeric_data.empty:
                    print(f"站点 {site_id} 没有数值数据")
                    continue
                
                # 计算相关性矩阵
                corr_matrix = numeric_data.corr()
                
                # 创建热图
                sns.heatmap(corr_matrix.round(2), annot=True, cmap='coolwarm', center=0,
                           ax=axes[row, col], square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                axes[row, col].set_title(f'{station_type} site {site_id}', fontsize=14, fontweight='bold')
            
            # 隐藏多余的子图
            total_sites = len(datasets)
            total_subplots = grid_shape[0] * grid_shape[1]
            for i in range(total_sites, total_subplots):
                row = i // grid_shape[1]
                col = i % grid_shape[1]
                axes[row, col].set_visible(False)
            
            plt.suptitle(f'{station_type} Correlation Analysis', fontsize=20, fontweight='bold')
            filename = f'{station_type}_correlation.png'
            plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{station_type} 相关性热图已保存为 {filename}")
            
        except Exception as e:
            print(f"创建 {station_type} 相关性热图失败: {e}")
    
    def generate_summary_report(self, datasets: Dict[int, pd.DataFrame], station_type: str):
        """生成数据摘要报告"""
        print(f"\n=== {station_type} 数据摘要报告 ===")
        
        total_capacity = 0
        total_avg_power = 0
        
        for site_id, df in datasets.items():
            print(f"\n站点 {site_id}:")
            print(f"  数据行数: {len(df)}")
            print(f"  数据列数: {len(df.columns)}")
            print(f"  缺失值总数: {df.isnull().sum().sum()}")
            
            # 数值列统计
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"  数值列数量: {len(numeric_cols)}")
                
                if 'Power(MW)' in df.columns:
                    avg_power = df['Power(MW)'].mean()
                    max_power = df['Power(MW)'].max()
                    print(f"  功率均值: {avg_power:.2f} MW")
                    print(f"  功率峰值: {max_power:.2f} MW")
                    total_avg_power += avg_power
                
                if 'Capacity(MW)' in df.columns:
                    capacity = df['Capacity(MW)'].iloc[0]
                    total_capacity += capacity
                    print(f"  装机容量: {capacity} MW")
                
                if 'Capacity_Factor(%)' in df.columns:
                    avg_cf = df['Capacity_Factor(%)'].mean()
                    print(f"  平均容量因子: {avg_cf:.2f}%")
        
        print(f"\n=== {station_type} 总体统计 ===")
        print(f"总装机容量: {total_capacity} MW")
        print(f"平均总功率: {total_avg_power:.2f} MW")
        if total_capacity > 0:
            print(f"总体容量因子: {(total_avg_power/total_capacity)*100:.2f}%")
    
    def run_comprehensive_analysis(self):
        """运行全面分析"""
        print("开始可再生能源数据全面分析...")
        
        # 加载风电场数据
        print("\n=== 加载风电场数据 ===")
        wind_datasets = self.load_and_process_data(self.wind_farm_config, "wind_farms")
        
        if wind_datasets:
            print("\n=== 风电场分析 ===")
            # 基础相关性分析
            self.create_correlation_heatmaps(wind_datasets, "Wind_Farm", (3, 2))
            
            # 时间序列分析
            self.create_time_series_analysis(wind_datasets, "Wind_Farm")
            
            # 功率分析
            self.create_power_analysis(wind_datasets, "Wind_Farm")
            
            # 气象条件分析
            self.create_meteorological_analysis(wind_datasets, "Wind_Farm")
            
            # 生成摘要报告
            self.generate_summary_report(wind_datasets, "风电场")
        
        # 加载太阳能电站数据
        print("\n=== 加载太阳能电站数据 ===")
        solar_datasets = self.load_and_process_data(self.solar_station_config, "solar_stations")
        
        if solar_datasets:
            print("\n=== 太阳能电站分析 ===")
            # 基础相关性分析
            self.create_correlation_heatmaps(solar_datasets, "Solar_Station", (4, 2))
            
            # 时间序列分析
            self.create_time_series_analysis(solar_datasets, "Solar_Station")
            
            # 功率分析
            self.create_power_analysis(solar_datasets, "Solar_Station")
            
            # 气象条件分析
            self.create_meteorological_analysis(solar_datasets, "Solar_Station")
            
            # 生成摘要报告
            self.generate_summary_report(solar_datasets, "太阳能电站")
        
        # 比较分析
        if wind_datasets and solar_datasets:
            print("\n=== 比较分析 ===")
            self.create_comparative_analysis(wind_datasets, solar_datasets)
        
        print(f"\n全面分析完成！所有结果已保存到 {self.figures_dir} 目录")


def main():
    """主函数"""
    try:
        analyzer = EnhancedRenewableEnergyAnalyzer()
        analyzer.run_comprehensive_analysis()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")


if __name__ == "__main__":
    main()