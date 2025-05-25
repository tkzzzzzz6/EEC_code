# PVOD数据集增强可视化 - 地图版本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import folium
from folium import plugins
import matplotlib.font_manager as fm
import sys
import os

# 设置警告
warnings.filterwarnings('ignore')

class PVODEnhancedVisualizer:
    """PVOD光伏数据集增强可视化器"""
    
    def __init__(self, data_dir: str = "PVODdatasets_v1.0", figures_dir: str = "Figures"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # 配置中文字体
        self.setup_chinese_fonts()
        
    def setup_chinese_fonts(self):
        """配置中文字体"""
        try:
            # 尝试设置多种中文字体
            chinese_fonts = [
                'SimHei',           # 黑体
                'Microsoft YaHei',  # 微软雅黑
                'SimSun',          # 宋体
                'KaiTi',           # 楷体
                'FangSong',        # 仿宋
                'Arial Unicode MS', # Arial Unicode MS
                'DejaVu Sans'      # DejaVu Sans (fallback)
            ]
            
            # 查找可用的中文字体
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            selected_font = None
            for font in chinese_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 已配置中文字体: {selected_font}")
            else:
                # 如果没有找到中文字体，尝试下载并安装
                self.download_chinese_font()
                
        except Exception as e:
            print(f"⚠️ 中文字体配置失败: {e}")
            # 使用默认设置
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
    
    def download_chinese_font(self):
        """下载中文字体"""
        try:
            import urllib.request
            import zipfile
            
            print("正在下载中文字体...")
            
            # 创建字体目录
            font_dir = Path("fonts")
            font_dir.mkdir(exist_ok=True)
            
            # 下载字体文件
            font_url = "https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSans.ttc"
            font_path = font_dir / "SourceHanSans.ttc"
            
            if not font_path.exists():
                urllib.request.urlretrieve(font_url, font_path)
                print("✓ 字体下载完成")
            
            # 注册字体
            fm.fontManager.addfont(str(font_path))
            plt.rcParams['font.sans-serif'] = ['Source Han Sans'] + ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            print("✓ 中文字体安装成功")
            
        except Exception as e:
            print(f"⚠️ 字体下载失败: {e}")
            # 使用系统默认字体
            if sys.platform.startswith('win'):
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
            else:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self):
        """加载数据"""
        print("正在加载PVOD数据...")
        
        # 加载元数据
        metadata = pd.read_csv(self.data_dir / "metadata.csv")
        
        # 加载所有站点数据
        station_data = {}
        for _, row in metadata.iterrows():
            station_id = row['Station_ID']
            try:
                df = pd.read_csv(self.data_dir / f"{station_id}.csv")
                
                # 数据处理
                df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
                for col in df.columns:
                    if col != 'date_time':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 添加站点信息
                capacity_kw = row['Capacity']
                capacity_mw = capacity_kw / 1000
                df['capacity_mw'] = capacity_mw
                df['capacity_factor'] = (df['power'] / capacity_mw) * 100
                df['technology'] = row['PV_Technology']
                df['longitude'] = row['Longitude']
                df['latitude'] = row['Latitude']
                
                station_data[station_id] = df
                print(f"✓ {station_id} 数据加载成功")
                
            except Exception as e:
                print(f"✗ {station_id} 数据加载失败: {e}")
        
        return station_data, metadata
    
    def create_interactive_map(self, station_data: Dict, metadata: pd.DataFrame):
        """创建交互式地图"""
        print("创建交互式地图...")
        
        # 计算地图中心
        center_lat = metadata['Latitude'].mean()
        center_lon = metadata['Longitude'].mean()
        
        # 创建基础地图
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        # 添加不同的地图图层
        folium.TileLayer('Stamen Terrain', name='地形图').add_to(m)
        folium.TileLayer('Stamen Toner', name='黑白图').add_to(m)
        folium.TileLayer('CartoDB positron', name='简洁图').add_to(m)
        
        # 准备数据
        performance_data = []
        for station_id, df in station_data.items():
            avg_cf = df['capacity_factor'].mean()
            max_power = df['power'].max()
            capacity = df['capacity_mw'].iloc[0]
            technology = df['technology'].iloc[0]
            lon = df['longitude'].iloc[0]
            lat = df['latitude'].iloc[0]
            
            performance_data.append({
                'station_id': station_id,
                'latitude': lat,
                'longitude': lon,
                'avg_cf': avg_cf,
                'max_power': max_power,
                'capacity': capacity,
                'technology': technology
            })
        
        # 创建性能数据DataFrame
        perf_df = pd.DataFrame(performance_data)
        
        # 按容量因子添加圆圈标记
        for _, row in perf_df.iterrows():
            # 根据容量因子确定圆圈大小（5-50像素）
            radius = max(5, min(50, row['avg_cf'] * 2))
            
            # 根据容量因子确定颜色
            if row['avg_cf'] > 20:
                color = 'darkgreen'
                fillColor = 'green'
            elif row['avg_cf'] > 15:
                color = 'orange'
                fillColor = 'yellow'
            elif row['avg_cf'] > 10:
                color = 'darkorange'
                fillColor = 'orange'
            else:
                color = 'darkred'
                fillColor = 'red'
            
            # 创建弹窗内容
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 250px;">
                <h4 style="margin-bottom: 10px; color: #2E86AB;">📍 {row['station_id']}</h4>
                <hr style="margin: 5px 0;">
                <p><strong>🔋 装机容量:</strong> {row['capacity']:.1f} MW</p>
                <p><strong>⚡ 平均容量因子:</strong> {row['avg_cf']:.1f}%</p>
                <p><strong>🔥 最大功率:</strong> {row['max_power']:.1f} MW</p>
                <p><strong>🔬 技术类型:</strong> {row['technology']}</p>
                <p><strong>📍 坐标:</strong> ({row['latitude']:.2f}°, {row['longitude']:.2f}°)</p>
            </div>
            """
            
            # 添加圆圈标记
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fillColor=fillColor,
                fillOpacity=0.7,
                weight=2,
                tooltip=f"{row['station_id']}: {row['avg_cf']:.1f}%"
            ).add_to(m)
            
            # 添加站点标签
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(
                    html=f"""<div style="
                        font-size: 10px; 
                        color: white; 
                        font-weight: bold; 
                        text-align: center;
                        text-shadow: 1px 1px 2px black;
                        background: rgba(0,0,0,0.5);
                        border-radius: 3px;
                        padding: 2px;
                    ">{row['station_id']}</div>""",
                    icon_size=(50, 20),
                    icon_anchor=(25, 10)
                )
            ).add_to(m)
        
        # 添加图例
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4 style="margin-top:0;">📊 容量因子图例</h4>
        <p><span style="color:green;">●</span> >20%: 优秀</p>
        <p><span style="color:orange;">●</span> 15-20%: 良好</p>
        <p><span style="color:darkorange;">●</span> 10-15%: 一般</p>
        <p><span style="color:red;">●</span> <10%: 较差</p>
        <p style="font-size:12px;"><em>圆圈大小代表性能水平</em></p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 添加热力图图层
        heat_data = [[row['latitude'], row['longitude'], row['avg_cf']] for _, row in perf_df.iterrows()]
        
        heat_map = plugins.HeatMap(
            heat_data,
            name='性能热力图',
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        )
        heat_map.add_to(m)
        
        # 添加聚类标记
        marker_cluster = plugins.MarkerCluster(name='聚类标记').add_to(m)
        
        for _, row in perf_df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"{row['station_id']}: {row['avg_cf']:.1f}%",
                icon=folium.Icon(color='green' if row['avg_cf'] > 15 else 'orange' if row['avg_cf'] > 10 else 'red')
            ).add_to(marker_cluster)
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        # 保存地图
        map_path = self.figures_dir / 'pvod_interactive_map.html'
        m.save(str(map_path))
        print(f"✓ 交互式地图已保存: {map_path}")
        
        return m
    
    def create_enhanced_geographic_plot(self, station_data: Dict, metadata: pd.DataFrame):
        """创建增强的地理分布图"""
        print("创建增强地理分布图...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('PVOD光伏站点地理分布分析', fontsize=16, fontweight='bold')
        
        # 准备数据
        lons = metadata['Longitude'].values
        lats = metadata['Latitude'].values
        capacities = metadata['Capacity'].values / 1000  # 转换为MW
        
        # 计算性能指标
        avg_cfs = []
        max_powers = []
        technologies = []
        
        for _, row in metadata.iterrows():
            station_id = row['Station_ID']
            if station_id in station_data:
                df = station_data[station_id]
                avg_cfs.append(df['capacity_factor'].mean())
                max_powers.append(df['power'].max())
                technologies.append(df['technology'].iloc[0])
            else:
                avg_cfs.append(0)
                max_powers.append(0)
                technologies.append('Unknown')
        
        # 1. 容量因子地理分布
        scatter1 = axes[0, 0].scatter(lons, lats, s=np.array(avg_cfs)*10, 
                                     c=avg_cfs, cmap='RdYlGn', alpha=0.7, edgecolors='black')
        axes[0, 0].set_title('按容量因子分布', fontweight='bold')
        axes[0, 0].set_xlabel('经度 (°)')
        axes[0, 0].set_ylabel('纬度 (°)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('平均容量因子 (%)')
        
        # 添加站点标签
        for i, station_id in enumerate(metadata['Station_ID']):
            axes[0, 0].annotate(station_id, (lons[i], lats[i]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.8)
        
        # 2. 装机容量地理分布
        scatter2 = axes[0, 1].scatter(lons, lats, s=capacities*20, 
                                     c=capacities, cmap='viridis', alpha=0.7, edgecolors='black')
        axes[0, 1].set_title('按装机容量分布', fontweight='bold')
        axes[0, 1].set_xlabel('经度 (°)')
        axes[0, 1].set_ylabel('纬度 (°)')
        axes[0, 1].grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
        cbar2.set_label('装机容量 (MW)')
        
        for i, station_id in enumerate(metadata['Station_ID']):
            axes[0, 1].annotate(station_id, (lons[i], lats[i]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.8)
        
        # 3. 技术类型分布
        tech_colors = {'Poly-Si': 'red', 'Mono-Si': 'blue'}
        for i, tech in enumerate(technologies):
            color = tech_colors.get(tech, 'gray')
            axes[1, 0].scatter(lons[i], lats[i], s=capacities[i]*20, 
                             c=color, alpha=0.7, edgecolors='black', 
                             label=tech if tech not in [t.get_text() for t in axes[1, 0].get_legend_handles_labels()[1]] else "")
        
        axes[1, 0].set_title('按技术类型分布', fontweight='bold')
        axes[1, 0].set_xlabel('经度 (°)')
        axes[1, 0].set_ylabel('纬度 (°)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        for i, station_id in enumerate(metadata['Station_ID']):
            axes[1, 0].annotate(station_id, (lons[i], lats[i]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.8)
        
        # 4. 性能vs地理位置相关性
        # 计算与中心点的距离
        center_lon, center_lat = np.mean(lons), np.mean(lats)
        distances = np.sqrt((lons - center_lon)**2 + (lats - center_lat)**2)
        
        scatter4 = axes[1, 1].scatter(distances, avg_cfs, s=capacities*20, 
                                     c=avg_cfs, cmap='RdYlGn', alpha=0.7, edgecolors='black')
        axes[1, 1].set_title('距离中心点vs性能', fontweight='bold')
        axes[1, 1].set_xlabel('距离地理中心距离 (度)')
        axes[1, 1].set_ylabel('平均容量因子 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(distances, avg_cfs, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(distances, p(distances), "r--", alpha=0.8, linewidth=2)
        
        # 计算相关系数
        corr = np.corrcoef(distances, avg_cfs)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'相关系数: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        for i, station_id in enumerate(metadata['Station_ID']):
            axes[1, 1].annotate(station_id, (distances[i], avg_cfs[i]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pvod_enhanced_geographic_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_performance_dashboard(self, station_data: Dict, metadata: pd.DataFrame):
        """创建性能仪表板"""
        print("创建性能仪表板...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('PVOD光伏电站性能综合仪表板', fontsize=16, fontweight='bold')
        
        # 准备数据
        station_stats = []
        for station_id, df in station_data.items():
            stats = {
                'station_id': station_id,
                'capacity_mw': df['capacity_mw'].iloc[0],
                'avg_cf': df['capacity_factor'].mean(),
                'max_cf': df['capacity_factor'].max(),
                'avg_power': df['power'].mean(),
                'max_power': df['power'].max(),
                'technology': df['technology'].iloc[0],
                'longitude': df['longitude'].iloc[0],
                'latitude': df['latitude'].iloc[0],
                'data_count': len(df),
                'operating_hours': (df['power'] > 0).sum(),
                'operating_ratio': (df['power'] > 0).sum() / len(df) * 100
            }
            station_stats.append(stats)
        
        stats_df = pd.DataFrame(station_stats)
        
        # 1. 容量因子排名
        sorted_stats = stats_df.sort_values('avg_cf', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_stats)))
        bars1 = axes[0, 0].barh(range(len(sorted_stats)), sorted_stats['avg_cf'], color=colors)
        axes[0, 0].set_title('各站点容量因子排名', fontweight='bold')
        axes[0, 0].set_xlabel('平均容量因子 (%)')
        axes[0, 0].set_yticks(range(len(sorted_stats)))
        axes[0, 0].set_yticklabels(sorted_stats['station_id'])
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, sorted_stats['avg_cf'])):
            axes[0, 0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                           f'{val:.1f}%', va='center', fontsize=9)
        
        # 2. 容量vs性能散点图
        scatter2 = axes[0, 1].scatter(stats_df['capacity_mw'], stats_df['avg_cf'], 
                                     s=stats_df['max_power']*5, alpha=0.7, 
                                     c=stats_df['avg_cf'], cmap='RdYlGn', edgecolors='black')
        axes[0, 1].set_title('装机容量vs平均容量因子', fontweight='bold')
        axes[0, 1].set_xlabel('装机容量 (MW)')
        axes[0, 1].set_ylabel('平均容量因子 (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        for _, row in stats_df.iterrows():
            axes[0, 1].annotate(row['station_id'], 
                              (row['capacity_mw'], row['avg_cf']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. 技术类型性能比较
        tech_perf = stats_df.groupby('technology')['avg_cf'].agg(['mean', 'std', 'count']).reset_index()
        bars3 = axes[0, 2].bar(tech_perf['technology'], tech_perf['mean'], 
                              yerr=tech_perf['std'], capsize=5, alpha=0.7, 
                              color=['lightblue', 'lightcoral'])
        axes[0, 2].set_title('技术类型性能比较', fontweight='bold')
        axes[0, 2].set_xlabel('技术类型')
        axes[0, 2].set_ylabel('平均容量因子 (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        for bar, val in zip(bars3, tech_perf['mean']):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{val:.1f}%', ha='center', fontsize=10)
        
        # 4. 运行时间分析
        bars4 = axes[1, 0].bar(stats_df['station_id'], stats_df['operating_ratio'], 
                              alpha=0.7, color='skyblue')
        axes[1, 0].set_title('各站点运行时间比例', fontweight='bold')
        axes[1, 0].set_xlabel('站点')
        axes[1, 0].set_ylabel('运行时间比例 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 最大功率vs平均功率
        axes[1, 1].scatter(stats_df['avg_power'], stats_df['max_power'], 
                          s=stats_df['capacity_mw']*10, alpha=0.7, 
                          c=stats_df['avg_cf'], cmap='RdYlGn', edgecolors='black')
        axes[1, 1].set_title('平均功率vs最大功率', fontweight='bold')
        axes[1, 1].set_xlabel('平均功率 (MW)')
        axes[1, 1].set_ylabel('最大功率 (MW)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加对角线
        max_val = max(stats_df['max_power'].max(), stats_df['avg_power'].max())
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='理论最大值')
        axes[1, 1].legend()
        
        # 6. 数据量分布
        bars6 = axes[1, 2].bar(stats_df['station_id'], stats_df['data_count'], 
                              alpha=0.7, color='lightgreen')
        axes[1, 2].set_title('各站点数据量', fontweight='bold')
        axes[1, 2].set_xlabel('站点')
        axes[1, 2].set_ylabel('数据记录数')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 性能分布直方图
        axes[2, 0].hist(stats_df['avg_cf'], bins=8, alpha=0.7, color='orange', edgecolor='black')
        axes[2, 0].axvline(stats_df['avg_cf'].mean(), color='red', linestyle='--', 
                          label=f"平均值: {stats_df['avg_cf'].mean():.1f}%")
        axes[2, 0].set_title('容量因子分布', fontweight='bold')
        axes[2, 0].set_xlabel('容量因子 (%)')
        axes[2, 0].set_ylabel('站点数量')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 地理位置聚类分析
        from sklearn.cluster import KMeans
        coords = stats_df[['longitude', 'latitude']].values
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(coords)
        
        colors_cluster = ['red', 'blue', 'green']
        for i in range(3):
            mask = clusters == i
            axes[2, 1].scatter(stats_df.loc[mask, 'longitude'], 
                             stats_df.loc[mask, 'latitude'], 
                             c=colors_cluster[i], s=100, alpha=0.7, 
                             label=f'区域 {i+1}')
        
        axes[2, 1].set_title('地理位置聚类分析', fontweight='bold')
        axes[2, 1].set_xlabel('经度 (°)')
        axes[2, 1].set_ylabel('纬度 (°)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. 综合性能雷达图
        # 选择前5个性能最好的站点
        top5_stations = stats_df.nlargest(5, 'avg_cf')
        
        categories = ['容量因子', '最大功率', '运行时间', '装机容量']
        
        # 标准化数据到0-1范围
        normalized_data = []
        for _, station in top5_stations.iterrows():
            values = [
                station['avg_cf'] / stats_df['avg_cf'].max(),
                station['max_power'] / stats_df['max_power'].max(),
                station['operating_ratio'] / stats_df['operating_ratio'].max(),
                station['capacity_mw'] / stats_df['capacity_mw'].max()
            ]
            normalized_data.append(values)
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 完成圆圈
        
        ax_radar = plt.subplot(3, 3, 9, projection='polar')
        
        colors_radar = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (_, station) in enumerate(top5_stations.iterrows()):
            values = normalized_data[i] + [normalized_data[i][0]]  # 完成圆圈
            ax_radar.plot(angles, values, 'o-', linewidth=2, 
                         label=station['station_id'], color=colors_radar[i])
            ax_radar.fill(angles, values, alpha=0.25, color=colors_radar[i])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('TOP5站点综合性能对比', fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax_radar.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pvod_performance_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def run_enhanced_analysis(self):
        """运行增强分析"""
        print("开始PVOD数据集增强可视化分析...")
        
        # 加载数据
        station_data, metadata = self.load_data()
        
        if not station_data:
            print("❌ 未找到有效数据，分析终止")
            return
        
        print(f"✅ 成功加载 {len(station_data)} 个站点的数据")
        
        # 创建各种可视化
        self.create_interactive_map(station_data, metadata)
        self.create_enhanced_geographic_plot(station_data, metadata)
        self.create_performance_dashboard(station_data, metadata)
        
        print("\n" + "="*60)
        print("🎉 增强可视化分析完成！")
        print("="*60)
        print("📁 生成的文件:")
        print("  📍 pvod_interactive_map.html - 交互式地图")
        print("  📊 pvod_enhanced_geographic_analysis.png - 增强地理分析")
        print("  📈 pvod_performance_dashboard.png - 性能仪表板")
        print("="*60)


def main():
    """主函数"""
    try:
        visualizer = PVODEnhancedVisualizer()
        visualizer.run_enhanced_analysis()
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 