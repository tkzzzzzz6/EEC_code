# PVOD数据集地图可视化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import folium
from folium import plugins
import matplotlib.font_manager as fm
import sys

# 设置警告
warnings.filterwarnings('ignore')

class PVODMapVisualizer:
    """PVOD光伏数据集地图可视化器"""
    
    def __init__(self, data_dir: str = "PVODdatasets_v1.0", figures_dir: str = "Figures"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # 配置中文字体
        self.setup_chinese_fonts()
        
    def setup_chinese_fonts(self):
        """配置中文字体"""
        try:
            # Windows系统的中文字体设置
            if sys.platform.startswith('win'):
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
            else:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
            plt.rcParams['axes.unicode_minus'] = False
            print("✓ 中文字体配置完成")
            
        except Exception as e:
            print(f"⚠️ 中文字体配置失败: {e}")
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
    
    def create_interactive_map(self, station_data: dict, metadata: pd.DataFrame):
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
        folium.TileLayer('CartoDB positron', name='简洁地图').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='暗色地图').add_to(m)
        
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
            # 根据容量因子确定圆圈大小（8-40像素）
            radius = max(8, min(40, row['avg_cf'] * 1.8))
            
            # 根据容量因子确定颜色
            if row['avg_cf'] > 20:
                color = 'darkgreen'
                fillColor = 'green'
                performance_level = '优秀'
            elif row['avg_cf'] > 15:
                color = 'orange'
                fillColor = 'yellow'
                performance_level = '良好'
            elif row['avg_cf'] > 10:
                color = 'darkorange'
                fillColor = 'orange'
                performance_level = '一般'
            else:
                color = 'darkred'
                fillColor = 'red'
                performance_level = '较差'
            
            # 创建弹窗内容
            popup_html = f"""
            <div style="font-family: 'Microsoft YaHei', Arial, sans-serif; width: 280px; padding: 10px;">
                <h3 style="margin: 0 0 10px 0; color: #2E86AB; text-align: center;">🔆 {row['station_id']}</h3>
                <hr style="margin: 10px 0; border: 1px solid #ddd;">
                
                <div style="margin: 8px 0;">
                    <span style="font-weight: bold; color: #333;">📍 地理位置:</span> 
                    <span style="color: #666;">({row['latitude']:.3f}°, {row['longitude']:.3f}°)</span>
                </div>
                
                <div style="margin: 8px 0;">
                    <span style="font-weight: bold; color: #333;">🔋 装机容量:</span> 
                    <span style="color: #e74c3c; font-weight: bold;">{row['capacity']:.1f} MW</span>
                </div>
                
                <div style="margin: 8px 0;">
                    <span style="font-weight: bold; color: #333;">⚡ 容量因子:</span> 
                    <span style="color: #27ae60; font-weight: bold;">{row['avg_cf']:.1f}%</span>
                    <span style="background: {fillColor}; color: white; padding: 2px 6px; border-radius: 10px; 
                           font-size: 11px; margin-left: 5px;">{performance_level}</span>
                </div>
                
                <div style="margin: 8px 0;">
                    <span style="font-weight: bold; color: #333;">🔥 最大功率:</span> 
                    <span style="color: #f39c12; font-weight: bold;">{row['max_power']:.1f} MW</span>
                </div>
                
                <div style="margin: 8px 0;">
                    <span style="font-weight: bold; color: #333;">🔬 技术类型:</span> 
                    <span style="color: #8e44ad;">{row['technology']}</span>
                </div>
                
                <div style="margin-top: 10px; padding: 5px; background: #f8f9fa; border-radius: 5px; font-size: 12px;">
                    <strong>性能等级:</strong> 圆圈大小表示容量因子水平
                </div>
            </div>
            """
            
            # 添加圆圈标记
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=320),
                color=color,
                fillColor=fillColor,
                fillOpacity=0.8,
                weight=3,
                tooltip=f"{row['station_id']}: {row['avg_cf']:.1f}% 容量因子"
            ).add_to(m)
            
            # 添加站点标签
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(
                    html=f"""<div style="
                        font-size: 11px; 
                        color: white; 
                        font-weight: bold; 
                        text-align: center;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
                        background: rgba(0,0,0,0.7);
                        border-radius: 4px;
                        padding: 3px 6px;
                        border: 1px solid white;
                    ">{row['station_id']}</div>""",
                    icon_size=(60, 25),
                    icon_anchor=(30, 12)
                )
            ).add_to(m)
        
        # 添加图例
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 160px; 
                    background-color: white; border: 2px solid #999; z-index: 9999; 
                    font-size: 13px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50; text-align: center; 
                       border-bottom: 1px solid #bdc3c7; padding-bottom: 5px;">
            </h4>
            <div style="margin: 8px 0;">
                <span style="color: green; font-size: 16px;">●</span> 
                <span style="margin-left: 8px;">>20%: 优秀表现</span>
            </div>
            <div style="margin: 8px 0;">
                <span style="color: orange; font-size: 16px;">●</span> 
                <span style="margin-left: 8px;">15-20%: 良好表现</span>
            </div>
            <div style="margin: 8px 0;">
                <span style="color: darkorange; font-size: 16px;">●</span> 
                <span style="margin-left: 8px;">10-15%: 一般表现</span>
            </div>
            <div style="margin: 8px 0;">
                <span style="color: red; font-size: 16px;">●</span> 
                <span style="margin-left: 8px;"><10%: 待提升</span>
            </div>
            <div style="margin-top: 10px; font-size: 11px; color: #7f8c8d; font-style: italic;">
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 添加热力图图层
        heat_data = [[row['latitude'], row['longitude'], row['avg_cf']] for _, row in perf_df.iterrows()]
        
        heat_map = plugins.HeatMap(
            heat_data,
            name='🔥 性能热力图',
            min_opacity=0.3,
            max_zoom=18,
            radius=30,
            blur=20,
            gradient={0.2: '#313695', 0.4: '#4575b4', 0.6: '#74add1', 0.8: '#fdae61', 1.0: '#d73027'}
        )
        heat_map.add_to(m)
        
        # 添加聚类标记
        marker_cluster = plugins.MarkerCluster(
            name='📍 聚类标记',
            options={'maxClusterRadius': 50}
        ).add_to(m)
        
        for _, row in perf_df.iterrows():
            icon_color = 'green' if row['avg_cf'] > 15 else 'orange' if row['avg_cf'] > 10 else 'red'
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"<b>{row['station_id']}</b><br>容量因子: {row['avg_cf']:.1f}%<br>装机容量: {row['capacity']:.1f}MW",
                icon=folium.Icon(color=icon_color, icon='bolt', prefix='fa')
            ).add_to(marker_cluster)
        
        # 添加距离测量工具
        plugins.MeasureControl().add_to(m)
        
        # 添加全屏控件
        plugins.Fullscreen().add_to(m)
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        # 保存地图
        map_path = self.figures_dir / 'pvod_interactive_map.html'
        m.save(str(map_path))
        print(f"✓ 交互式地图已保存: {map_path}")
        
        return m
    
    def create_enhanced_geographic_plots(self, station_data: dict, metadata: pd.DataFrame):
        """创建增强的地理分布图"""
        print("创建增强地理分布静态图...")
        
        # 设置中文字体和样式
        plt.rcParams['font.size'] = 10
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('PVOD光伏站点地理分布深度分析', fontsize=16, fontweight='bold', y=0.95)
        
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
        
        # 1. 容量因子地理分布（气泡图）
        scatter1 = axes[0, 0].scatter(lons, lats, s=np.array(avg_cfs)*12, 
                                     c=avg_cfs, cmap='RdYlGn', alpha=0.8, 
                                     edgecolors='black', linewidth=1.5)
        axes[0, 0].set_title('各站点容量因子地理分布', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('经度 (°)')
        axes[0, 0].set_ylabel('纬度 (°)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0], shrink=0.8)
        cbar1.set_label('平均容量因子 (%)', fontsize=10)
        
        # 添加站点标签和数值
        for i, (station_id, cf) in enumerate(zip(metadata['Station_ID'], avg_cfs)):
            axes[0, 0].annotate(f'{station_id}\n{cf:.1f}%', 
                              (lons[i], lats[i]), 
                              xytext=(8, 8), textcoords='offset points', 
                              fontsize=8, ha='left',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # 2. 装机容量地理分布
        scatter2 = axes[0, 1].scatter(lons, lats, s=capacities*25, 
                                     c=capacities, cmap='plasma', alpha=0.8, 
                                     edgecolors='black', linewidth=1.5)
        axes[0, 1].set_title('各站点装机容量地理分布', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('经度 (°)')
        axes[0, 1].set_ylabel('纬度 (°)')
        axes[0, 1].grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('装机容量 (MW)', fontsize=10)
        
        for i, (station_id, cap) in enumerate(zip(metadata['Station_ID'], capacities)):
            axes[0, 1].annotate(f'{station_id}\n{cap:.1f}MW', 
                              (lons[i], lats[i]), 
                              xytext=(8, 8), textcoords='offset points', 
                              fontsize=8, ha='left',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # 3. 技术类型和性能组合分布
        tech_colors = {'Poly-Si': '#e74c3c', 'Mono-Si': '#3498db'}
        tech_markers = {'Poly-Si': 'o', 'Mono-Si': 's'}
        
        for tech in set(technologies):
            if tech != 'Unknown':
                tech_mask = [t == tech for t in technologies]
                tech_lons = np.array(lons)[tech_mask]
                tech_lats = np.array(lats)[tech_mask]
                tech_cfs = np.array(avg_cfs)[tech_mask]
                
                axes[1, 0].scatter(tech_lons, tech_lats, 
                                 s=tech_cfs*15, c=tech_colors[tech], 
                                 marker=tech_markers[tech], alpha=0.8,
                                 edgecolors='black', linewidth=1.5,
                                 label=f'{tech} (平均: {np.mean(tech_cfs):.1f}%)')
        
        axes[1, 0].set_title('技术类型分布及性能对比', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('经度 (°)')
        axes[1, 0].set_ylabel('纬度 (°)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc='upper right')
        
        # 4. 地理位置与性能相关性分析
        # 计算与最佳性能站点的距离
        best_idx = np.argmax(avg_cfs)
        best_lon, best_lat = lons[best_idx], lats[best_idx]
        
        distances = np.sqrt((lons - best_lon)**2 + (lats - best_lat)**2)
        
        scatter4 = axes[1, 1].scatter(distances, avg_cfs, s=capacities*20, 
                                     c=avg_cfs, cmap='RdYlGn', alpha=0.8, 
                                     edgecolors='black', linewidth=1.5)
        
        # 添加趋势线
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(distances, avg_cfs)
        line = slope * distances + intercept
        axes[1, 1].plot(distances, line, 'r--', alpha=0.8, linewidth=2, 
                       label=f'趋势线 (R^2 = {r_value**2:.3f})')
        
        axes[1, 1].set_title('距离最优站点位置与性能关系', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('距离最优站点距离 (度)')
        axes[1, 1].set_ylabel('平均容量因子 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # 添加站点标签
        for i, station_id in enumerate(metadata['Station_ID']):
            axes[1, 1].annotate(station_id, (distances[i], avg_cfs[i]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.8)
        
        # 在图中标注最优站点
        best_station = metadata['Station_ID'].iloc[best_idx]
        axes[1, 1].axhline(y=avg_cfs[best_idx], color='g', linestyle=':', alpha=0.7, 
                          label=f'最优性能: {best_station} ({avg_cfs[best_idx]:.1f}%)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pvod_enhanced_geographic_analysis_fixed.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✓ 增强地理分布图已保存")
    
    def run_analysis(self):
        """运行分析"""
        print("🚀 开始PVOD数据集地图可视化分析...")
        print("="*60)
        
        # 加载数据
        station_data, metadata = self.load_data()
        
        if not station_data:
            print("❌ 未找到有效数据，分析终止")
            return
        
        print(f"✅ 成功加载 {len(station_data)} 个站点的数据")
        print("="*60)
        
        # 创建各种可视化
        self.create_interactive_map(station_data, metadata)
        self.create_enhanced_geographic_plots(station_data, metadata)
        
        print("\n" + "="*60)
        print("🎉 地图可视化分析完成！")
        print("="*60)
        print("📁 生成的文件:")
        print("  🗺️  pvod_interactive_map.html - 交互式地图 (可在浏览器中打开)")
        print("  📊 pvod_enhanced_geographic_analysis_fixed.png - 静态地理分析图")
        print("="*60)
        print("💡 建议:")
        print("  1. 用浏览器打开HTML文件查看交互式地图")
        print("  2. 地图支持图层切换、热力图、聚类等功能")
        print("  3. 圆圈大小代表容量因子，颜色表示性能等级")
        print("="*60)


def main():
    """主函数"""
    try:
        visualizer = PVODMapVisualizer()
        visualizer.run_analysis()
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 