# 光伏发电功率预测综合分析系统
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import sys

# 导入自定义模块
from power_prediction_model import PowerPredictionModel
from prediction_visualization import PredictionVisualizer

warnings.filterwarnings('ignore')

class ComprehensivePredictionAnalysis:
    """综合预测分析系统"""
    
    def __init__(self, data_dir: str = "../PVODdatasets_v1.0"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.predictor = PowerPredictionModel(data_dir)
        self.visualizer = PredictionVisualizer()
        
    def analyze_single_station(self, station_id: str, forecast_days: int = 7):
        """分析单个站点"""
        print(f"\n🎯 开始分析站点: {station_id}")
        print("="*60)
        
        try:
            # 1. 加载数据
            print("📊 步骤1: 加载数据...")
            df = self.predictor.load_station_data(station_id)
            print(f"  数据范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
            print(f"  数据点数: {len(df):,}")
            print(f"  平均功率: {df['power'].mean():.3f} MW")
            print(f"  最大功率: {df['power'].max():.3f} MW")
            
            # 2. 特征工程
            print("\n🔧 步骤2: 特征工程...")
            df_features = self.predictor.prepare_features(df)
            print(f"  特征数量: {len(self.predictor.feature_names)}")
            print(f"  有效数据: {len(df_features):,} 条")
            
            # 3. 模型训练
            print("\n🚀 步骤3: 模型训练...")
            results = self.predictor.train_model(df_features)
            
            # 4. 未来预测
            print(f"\n🔮 步骤4: 预测未来 {forecast_days} 天...")
            forecast_df = self.predictor.predict_future(df, forecast_days=forecast_days)
            
            # 5. 保存结果
            print("\n💾 步骤5: 保存结果...")
            
            # 保存预测结果
            forecast_file = self.results_dir / f"{station_id}_7day_forecast.csv"
            forecast_df.to_csv(forecast_file, index=False)
            print(f"  预测结果: {forecast_file}")
            
            # 保存模型
            model_file = self.results_dir / f"{station_id}_xgboost_model.pkl"
            self.predictor.save_model(model_file)
            print(f"  模型文件: {model_file}")
            
            # 保存详细结果
            detailed_results = {
                'station_id': station_id,
                'data_points': len(df),
                'feature_count': len(self.predictor.feature_names),
                'train_size': len(results['X_test']) * 4,  # 估算训练集大小
                'test_size': len(results['X_test']),
                'train_metrics': results['train_metrics'],
                'test_metrics': results['test_metrics'],
                'forecast_points': len(forecast_df),
                'forecast_start': forecast_df['date_time'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'forecast_end': forecast_df['date_time'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'avg_predicted_power': forecast_df['predicted_power'].mean(),
                'max_predicted_power': forecast_df['predicted_power'].max(),
                'min_predicted_power': forecast_df['predicted_power'].min()
            }
            
            # 6. 生成可视化分析
            print("\n🎨 步骤6: 生成可视化分析...")
            self.visualizer.create_comprehensive_report(
                df, forecast_df, results, 
                self.predictor.model, self.predictor.feature_names, station_id
            )
            
            print(f"\n✅ {station_id} 分析完成！")
            return detailed_results
            
        except Exception as e:
            print(f"❌ {station_id} 分析失败: {str(e)}")
            return None
    
    def analyze_multiple_stations(self, station_ids: list, forecast_days: int = 7):
        """分析多个站点"""
        print(f"🌞 光伏发电功率预测综合分析系统")
        print(f"📅 预测天数: {forecast_days} 天")
        print(f"🎯 分析站点: {', '.join(station_ids)}")
        print("="*80)
        
        all_results = {}
        
        for i, station_id in enumerate(station_ids, 1):
            print(f"\n🔄 进度: {i}/{len(station_ids)}")
            result = self.analyze_single_station(station_id, forecast_days)
            if result:
                all_results[station_id] = result
        
        # 生成对比分析
        if len(all_results) > 1:
            self.generate_comparison_analysis(all_results)
        
        return all_results
    
    def generate_comparison_analysis(self, all_results: dict):
        """生成对比分析报告"""
        print(f"\n📊 生成多站点对比分析...")
        
        # 创建对比数据框
        comparison_data = []
        for station_id, result in all_results.items():
            comparison_data.append({
                '站点ID': station_id,
                '数据点数': result['data_points'],
                '特征数量': result['feature_count'],
                '测试集MAE': result['test_metrics']['mae'],
                '测试集RMSE': result['test_metrics']['rmse'],
                '测试集R²': result['test_metrics']['r2'],
                '测试集MAPE': result['test_metrics']['mape'],
                '预测平均功率': result['avg_predicted_power'],
                '预测最大功率': result['max_predicted_power'],
                '预测最小功率': result['min_predicted_power']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存对比结果
        comparison_file = self.results_dir / "stations_prediction_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        
        # 打印对比摘要
        print(f"\n📋 多站点预测性能对比:")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # 性能排名
        print(f"\n🏆 性能排名 (按R²值):")
        ranking = comparison_df.sort_values('测试集R²', ascending=False)
        for i, (_, row) in enumerate(ranking.iterrows(), 1):
            print(f"  {i}. {row['站点ID']}: R² = {row['测试集R²']:.4f}, MAE = {row['测试集MAE']:.4f}")
        
        print(f"\n💾 对比结果已保存到: {comparison_file}")
    
    def create_prediction_summary_report(self, all_results: dict):
        """创建预测摘要报告"""
        report_file = self.results_dir / "prediction_summary_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 光伏发电功率预测分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 分析概览\n\n")
            f.write(f"- **分析站点数**: {len(all_results)}\n")
            f.write(f"- **预测时长**: 7天 (672个15分钟时段)\n")
            f.write(f"- **模型类型**: XGBoost回归模型\n")
            f.write(f"- **特征类型**: 基于历史功率的时间序列特征\n\n")
            
            f.write("## 🎯 各站点预测性能\n\n")
            for station_id, result in all_results.items():
                f.write(f"### {station_id}\n\n")
                f.write(f"- **数据规模**: {result['data_points']:,} 个历史数据点\n")
                f.write(f"- **特征数量**: {result['feature_count']} 个\n")
                f.write(f"- **测试集性能**:\n")
                f.write(f"  - MAE: {result['test_metrics']['mae']:.4f} MW\n")
                f.write(f"  - RMSE: {result['test_metrics']['rmse']:.4f} MW\n")
                f.write(f"  - R²: {result['test_metrics']['r2']:.4f}\n")
                f.write(f"  - MAPE: {result['test_metrics']['mape']:.2f}%\n")
                f.write(f"- **预测结果**:\n")
                f.write(f"  - 平均预测功率: {result['avg_predicted_power']:.3f} MW\n")
                f.write(f"  - 最大预测功率: {result['max_predicted_power']:.3f} MW\n")
                f.write(f"  - 预测时间范围: {result['forecast_start']} 到 {result['forecast_end']}\n\n")
            
            f.write("## 📈 模型特点\n\n")
            f.write("### 特征工程\n")
            f.write("1. **时间特征**: 小时、分钟、星期、月份等周期性特征\n")
            f.write("2. **滞后特征**: 1-672个时段的历史功率值\n")
            f.write("3. **滚动统计**: 不同时间窗口的均值、标准差、最值\n")
            f.write("4. **同时段统计**: 历史同一时段的统计特征\n")
            f.write("5. **差分特征**: 不同时间间隔的功率变化\n\n")
            
            f.write("### 模型优势\n")
            f.write("- ✅ **纯历史功率驱动**: 仅使用历史功率数据，无需气象预报\n")
            f.write("- ✅ **高频预测**: 15分钟级别的精细化预测\n")
            f.write("- ✅ **长期预测**: 支持7天日前预测\n")
            f.write("- ✅ **自动特征工程**: 自动提取时间序列特征\n")
            f.write("- ✅ **鲁棒性强**: XGBoost模型对异常值不敏感\n\n")
            
            f.write("## 📁 输出文件\n\n")
            f.write("- `{station_id}_7day_forecast.csv`: 7天预测结果\n")
            f.write("- `{station_id}_xgboost_model.pkl`: 训练好的模型文件\n")
            f.write("- `{station_id}_*.png`: 可视化分析图表\n")
            f.write("- `stations_prediction_comparison.csv`: 多站点对比结果\n\n")
        
        print(f"📄 预测摘要报告已生成: {report_file}")


def main():
    """主函数"""
    print("🌞 光伏发电功率预测综合分析系统")
    print("="*60)
    
    # 初始化分析系统
    analyzer = ComprehensivePredictionAnalysis()
    
    # 选择要分析的站点
    station_ids = ["station01", "station04", "station09"]  # 可以根据需要修改
    
    # 执行综合分析
    all_results = analyzer.analyze_multiple_stations(station_ids, forecast_days=7)
    
    # 生成摘要报告
    if all_results:
        analyzer.create_prediction_summary_report(all_results)
    
    print(f"\n🎉 所有分析完成！")
    print(f"📁 结果文件保存在: {analyzer.results_dir}")
    print(f"📊 共分析了 {len(all_results)} 个站点")
    
    # 显示最终摘要
    if all_results:
        print(f"\n📋 最终摘要:")
        for station_id, result in all_results.items():
            r2 = result['test_metrics']['r2']
            mae = result['test_metrics']['mae']
            avg_power = result['avg_predicted_power']
            print(f"  {station_id}: R²={r2:.4f}, MAE={mae:.4f}MW, 平均预测功率={avg_power:.3f}MW")


if __name__ == "__main__":
    main() 