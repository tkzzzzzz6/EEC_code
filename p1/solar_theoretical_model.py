# 太阳辐照理论模型 - 光伏发电理论功率计算
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import pytz

# 简化的中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')

warnings.filterwarnings('ignore')

class SolarTheoreticalModel:
    """太阳辐照理论模型类"""
    
    def __init__(self):
        # 太阳常数 (W/m²)
        self.solar_constant = 1367.0
        
        # 地球轨道偏心率修正系数
        self.eccentricity_correction = {
            1: 1.033, 2: 1.031, 3: 1.026, 4: 1.018, 5: 1.009, 6: 1.002,
            7: 0.996, 8: 0.992, 9: 0.998, 10: 1.007, 11: 1.018, 12: 1.028
        }
        
        # 大气透射率参数
        self.atmospheric_transmittance = 0.75  # 晴天条件下的大气透射率
        
    def utc_to_beijing_time(self, utc_datetime: datetime) -> datetime:
        """
        将UTC时间转换为北京时间
        
        Args:
            utc_datetime: UTC时间
            
        Returns:
            北京时间
        """
        # 如果输入的datetime没有时区信息，假设它是UTC
        if utc_datetime.tzinfo is None:
            utc_datetime = pytz.UTC.localize(utc_datetime)
        
        # 转换为北京时间 (UTC+8)
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = utc_datetime.astimezone(beijing_tz)
        
        # 返回naive datetime（去掉时区信息）
        return beijing_time.replace(tzinfo=None)
        
    def calculate_solar_position(self, latitude: float, longitude: float, 
                               datetime_obj: datetime, is_utc: bool = True) -> Tuple[float, float]:
        """
        计算太阳位置角度
        
        Args:
            latitude: 纬度 (度)
            longitude: 经度 (度) 
            datetime_obj: 日期时间对象
            is_utc: 输入时间是否为UTC时间
            
        Returns:
            (太阳高度角, 太阳方位角) 单位：度
        """
        # 如果输入是UTC时间，转换为北京时间进行计算
        if is_utc:
            local_datetime = self.utc_to_beijing_time(datetime_obj)
        else:
            local_datetime = datetime_obj
            
        # 转换为弧度
        lat_rad = math.radians(latitude)
        
        # 计算一年中的第几天
        day_of_year = local_datetime.timetuple().tm_yday
        
        # 计算太阳赤纬角 (度)
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
        decl_rad = math.radians(declination)
        
        # 计算时角 (度) - 使用当地时间
        hour = local_datetime.hour + local_datetime.minute / 60.0
        # 考虑经度修正（相对于东经120度的时差）
        solar_time = hour + (longitude - 120) / 15
        hour_angle = 15 * (solar_time - 12)
        hour_angle_rad = math.radians(hour_angle)
        
        # 计算太阳高度角
        sin_elevation = (math.sin(lat_rad) * math.sin(decl_rad) + 
                        math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle_rad))
        elevation = math.degrees(math.asin(max(-1, min(1, sin_elevation))))
        
        # 计算太阳方位角
        if elevation > 0:  # 只有太阳在地平线以上时才计算方位角
            cos_azimuth = ((math.sin(decl_rad) * math.cos(lat_rad) - 
                           math.cos(decl_rad) * math.sin(lat_rad) * math.cos(hour_angle_rad)) / 
                          math.cos(math.radians(elevation)))
            cos_azimuth = max(-1, min(1, cos_azimuth))
            azimuth = math.degrees(math.acos(cos_azimuth))
            
            # 修正方位角象限
            if hour_angle > 0:
                azimuth = 360 - azimuth
        else:
            azimuth = 0  # 太阳在地平线以下时方位角设为0
            
        return elevation, azimuth
    
    def calculate_theoretical_irradiance(self, latitude: float, longitude: float,
                                       datetime_obj: datetime, 
                                       panel_tilt: float = 33.0,
                                       panel_azimuth: float = 180.0,
                                       is_utc: bool = True) -> Dict[str, float]:
        """
        计算理论太阳辐照度
        
        Args:
            latitude: 纬度
            longitude: 经度
            datetime_obj: 日期时间
            panel_tilt: 光伏板倾斜角 (度)
            panel_azimuth: 光伏板方位角 (度，南向为180)
            is_utc: 输入时间是否为UTC时间
            
        Returns:
            包含各种辐照度的字典
        """
        # 获取太阳位置
        solar_elevation, solar_azimuth = self.calculate_solar_position(
            latitude, longitude, datetime_obj, is_utc)
        
        # 如果太阳在地平线以下，辐照度为0
        if solar_elevation <= 0:
            return {
                'solar_elevation': solar_elevation,
                'solar_azimuth': solar_azimuth,
                'dni': 0.0,  # 直射辐照度
                'dhi': 0.0,  # 散射辐照度
                'ghi': 0.0,  # 全球水平辐照度
                'poa': 0.0   # 倾斜面辐照度
            }
        
        # 计算大气质量
        air_mass = 1 / math.sin(math.radians(solar_elevation))
        if air_mass > 10:
            air_mass = 10  # 限制最大大气质量
        
        # 计算地外辐照度
        # 使用当地时间确定月份
        if is_utc:
            local_datetime = self.utc_to_beijing_time(datetime_obj)
        else:
            local_datetime = datetime_obj
        month = local_datetime.month
        
        extraterrestrial_irradiance = (self.solar_constant * 
                                     self.eccentricity_correction[month])
        
        # 计算直射法向辐照度 (DNI)
        dni = (extraterrestrial_irradiance * 
               (self.atmospheric_transmittance ** air_mass))
        
        # 计算全球水平辐照度 (GHI)
        ghi = dni * math.sin(math.radians(solar_elevation))
        
        # 计算散射辐照度 (DHI) - 简化模型
        dhi = ghi * 0.15  # 假设散射占总辐照的15%
        
        # 计算倾斜面辐照度 (POA)
        poa = self.calculate_poa_irradiance(
            dni, dhi, ghi, solar_elevation, solar_azimuth,
            panel_tilt, panel_azimuth)
        
        return {
            'solar_elevation': solar_elevation,
            'solar_azimuth': solar_azimuth,
            'dni': dni,
            'dhi': dhi,
            'ghi': ghi,
            'poa': poa
        }
    
    def calculate_poa_irradiance(self, dni: float, dhi: float, ghi: float,
                               solar_elevation: float, solar_azimuth: float,
                               panel_tilt: float, panel_azimuth: float) -> float:
        """
        计算倾斜面辐照度 (Plane of Array Irradiance)
        
        Args:
            dni: 直射法向辐照度
            dhi: 散射辐照度
            ghi: 全球水平辐照度
            solar_elevation: 太阳高度角
            solar_azimuth: 太阳方位角
            panel_tilt: 光伏板倾斜角
            panel_azimuth: 光伏板方位角
            
        Returns:
            倾斜面辐照度
        """
        # 转换为弧度
        solar_elev_rad = math.radians(solar_elevation)
        solar_azim_rad = math.radians(solar_azimuth)
        panel_tilt_rad = math.radians(panel_tilt)
        panel_azim_rad = math.radians(panel_azimuth)
        
        # 计算太阳与光伏板法向的夹角余弦值
        cos_incidence = (math.sin(solar_elev_rad) * math.cos(panel_tilt_rad) +
                        math.cos(solar_elev_rad) * math.sin(panel_tilt_rad) *
                        math.cos(solar_azim_rad - panel_azim_rad))
        
        # 确保余弦值在有效范围内
        cos_incidence = max(0, cos_incidence)
        
        # 计算直射分量
        beam_component = dni * cos_incidence
        
        # 计算散射分量 (各向同性天空模型)
        diffuse_component = dhi * (1 + math.cos(panel_tilt_rad)) / 2
        
        # 计算地面反射分量 (假设地面反射率为0.2)
        ground_reflectance = 0.2
        reflected_component = (ghi * ground_reflectance * 
                             (1 - math.cos(panel_tilt_rad)) / 2)
        
        # 总倾斜面辐照度
        poa = beam_component + diffuse_component + reflected_component
        
        return max(0, poa)
    
    def calculate_theoretical_power(self, poa_irradiance: float, 
                                  capacity_kw: float,
                                  temperature: float = 25.0,
                                  efficiency_factor: float = 0.85) -> float:
        """
        计算理论发电功率
        
        Args:
            poa_irradiance: 倾斜面辐照度 (W/m²)
            capacity_kw: 装机容量 (kW)
            temperature: 环境温度 (°C)
            efficiency_factor: 系统效率因子
            
        Returns:
            理论功率 (MW)
        """
        # 标准测试条件下的辐照度
        stc_irradiance = 1000.0  # W/m²
        
        # 温度系数 (每度温度变化对功率的影响，通常为-0.4%/°C)
        temp_coefficient = -0.004
        stc_temperature = 25.0
        
        # 计算温度修正因子
        temp_factor = 1 + temp_coefficient * (temperature - stc_temperature)
        
        # 计算理论功率
        theoretical_power_kw = (capacity_kw * 
                              (poa_irradiance / stc_irradiance) * 
                              temp_factor * 
                              efficiency_factor)
        
        # 转换为MW并确保非负
        theoretical_power_mw = max(0, theoretical_power_kw / 1000.0)
        
        return theoretical_power_mw


class PVPerformanceAnalyzer:
    """光伏性能分析器"""
    
    def __init__(self, data_dir: str = "../PVODdatasets_v1.0"):
        self.data_dir = Path(data_dir)
        self.solar_model = SolarTheoreticalModel()
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> pd.DataFrame:
        """加载元数据"""
        metadata_path = self.data_dir / "metadata.csv"
        return pd.read_csv(metadata_path)
    
    def load_station_data(self, station_id: str) -> pd.DataFrame:
        """加载站点数据"""
        file_path = self.data_dir / f"{station_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        # 数据清理
        numeric_columns = [col for col in df.columns if col != 'date_time']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def calculate_theoretical_performance(self, station_id: str) -> pd.DataFrame:
        """计算站点的理论性能"""
        print(f"正在计算 {station_id} 的理论性能...")
        
        # 获取站点信息
        station_info = self.metadata[self.metadata['Station_ID'] == station_id].iloc[0]
        latitude = station_info['Latitude']
        longitude = station_info['Longitude']
        capacity_kw = station_info['Capacity']
        
        # 解析倾斜角度
        array_tilt_str = station_info['Array_Tilt']
        panel_tilt = float(array_tilt_str.split('°')[0].split()[-1])
        
        # 加载实际数据
        df = self.load_station_data(station_id)
        
        # 计算理论值
        theoretical_data = []
        
        for idx, row in df.iterrows():
            datetime_obj = row['date_time'].to_pydatetime()
            temperature = row.get('nwp_temperature', 25.0)
            if pd.isna(temperature):
                temperature = 25.0
            
            # 计算理论辐照度 - 明确指定输入时间为UTC
            irradiance_data = self.solar_model.calculate_theoretical_irradiance(
                latitude, longitude, datetime_obj, panel_tilt, is_utc=True)
            
            # 计算理论功率
            theoretical_power = self.solar_model.calculate_theoretical_power(
                irradiance_data['poa'], capacity_kw, temperature)
            
            theoretical_data.append({
                'date_time': datetime_obj,
                'theoretical_ghi': irradiance_data['ghi'],
                'theoretical_dni': irradiance_data['dni'],
                'theoretical_poa': irradiance_data['poa'],
                'theoretical_power': theoretical_power,
                'solar_elevation': irradiance_data['solar_elevation'],
                'solar_azimuth': irradiance_data['solar_azimuth']
            })
            
            if idx % 1000 == 0:
                print(f"  已处理 {idx}/{len(df)} 条记录...")
        
        # 合并理论数据和实际数据
        theoretical_df = pd.DataFrame(theoretical_data)
        result_df = df.merge(theoretical_df, on='date_time', how='left')
        
        # 计算性能指标
        result_df['power_ratio'] = result_df['power'] / result_df['theoretical_power']
        result_df['power_difference'] = result_df['power'] - result_df['theoretical_power']
        result_df['performance_ratio'] = result_df['power_ratio'].fillna(0)
        
        # 添加时间特征
        result_df['hour'] = result_df['date_time'].dt.hour
        result_df['month'] = result_df['date_time'].dt.month
        result_df['season'] = result_df['month'].map({
            12: '冬季', 1: '冬季', 2: '冬季',
            3: '春季', 4: '春季', 5: '春季',
            6: '夏季', 7: '夏季', 8: '夏季',
            9: '秋季', 10: '秋季', 11: '秋季'
        })
        
        print(f"✓ {station_id} 理论性能计算完成")
        return result_df
    
    def analyze_performance_patterns(self, df: pd.DataFrame, station_id: str) -> Dict:
        """分析性能模式 - 只考虑白昼时段"""
        # 过滤白昼数据：太阳高度角>0 且 理论功率>0 且 实际功率>=0
        daytime_df = df[
            (df['solar_elevation'] > 0) & 
            (df['theoretical_power'] > 0) & 
            (df['power'] >= 0)
        ].copy()
        
        if len(daytime_df) == 0:
            print(f"⚠️ {station_id} 没有有效的白昼数据")
            return {}
        
        print(f"📊 {station_id} 白昼数据统计:")
        print(f"  总记录数: {len(df):,}")
        print(f"  白昼记录数: {len(daytime_df):,} ({len(daytime_df)/len(df)*100:.1f}%)")
        
        # 计算统计指标
        stats = {
            'station_id': station_id,
            'total_records': len(df),
            'daytime_records': len(daytime_df),
            'mean_actual_power': daytime_df['power'].mean(),
            'mean_theoretical_power': daytime_df['theoretical_power'].mean(),
            'mean_performance_ratio': daytime_df['performance_ratio'].mean(),
            'std_performance_ratio': daytime_df['performance_ratio'].std(),
            'max_performance_ratio': daytime_df['performance_ratio'].max(),
            'min_performance_ratio': daytime_df['performance_ratio'].min(),
        }
        
        # 计算功率损失
        total_theoretical = daytime_df['theoretical_power'].sum()
        total_actual = daytime_df['power'].sum()
        power_loss = total_theoretical - total_actual
        power_loss_percent = (power_loss / total_theoretical) * 100 if total_theoretical > 0 else 0
        
        stats['total_power_loss_mw'] = power_loss
        stats['power_loss_percent'] = power_loss_percent
        
        # 季节性分析
        seasonal_stats = daytime_df.groupby('season').agg({
            'performance_ratio': ['mean', 'std'],
            'power': 'mean',
            'theoretical_power': 'mean'
        }).round(3)
        
        # 日内变化分析
        hourly_stats = daytime_df.groupby('hour').agg({
            'performance_ratio': ['mean', 'std'],
            'power': 'mean',
            'theoretical_power': 'mean'
        }).round(3)
        
        stats['seasonal_analysis'] = seasonal_stats
        stats['hourly_analysis'] = hourly_stats
        
        return stats


def main():
    """主函数"""
    print("🚀 开始光伏发电理论vs实际性能分析...")
    
    analyzer = PVPerformanceAnalyzer()
    
    # 选择一个站点进行详细分析 (可以修改为分析所有站点)
    station_id = "station01"  # 选择数据量较大的站点
    
    try:
        # 计算理论性能
        performance_df = analyzer.calculate_theoretical_performance(station_id)
        
        # 分析性能模式
        stats = analyzer.analyze_performance_patterns(performance_df, station_id)
        
        # 保存结果
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        performance_df.to_csv(output_dir / f"{station_id}_theoretical_vs_actual.csv", index=False)
        
        print(f"\n✅ 分析完成！")
        print(f"📊 性能统计:")
        print(f"  平均实际功率: {stats['mean_actual_power']:.3f} MW")
        print(f"  平均理论功率: {stats['mean_theoretical_power']:.3f} MW") 
        print(f"  平均性能比: {stats['mean_performance_ratio']:.3f}")
        print(f"  性能比标准差: {stats['std_performance_ratio']:.3f}")
        
        print(f"\n📁 结果已保存到: {output_dir}")
        
        return performance_df, stats
        
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    performance_df, stats = main() 