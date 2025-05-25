#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3: 融入NWP信息的光伏发电功率预测模型 - 一键运行脚本

这个脚本将运行完整的Problem 3分析，包括：
1. NWP信息融入的预测模型训练
2. 多模型对比分析
3. 评价指标计算
4. 可视化结果生成
5. 综合报告输出

使用方法:
    python run_problem3_analysis.py

作者: AI Assistant
日期: 2024年12月
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """打印标题"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def print_step(step_num, description):
    """打印步骤"""
    print(f"\n📋 步骤 {step_num}: {description}")
    print("-" * 60)

def run_analysis():
    """运行完整的Problem 3分析"""
    
    print_header("Problem 3: 融入NWP信息的光伏发电功率预测模型分析")
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📁 当前工作目录: {current_dir}")
    
    # 检查必要文件
    required_files = [
        'improved_power_prediction_with_nwp.py',
        'evaluation_metrics_nwp.py',
        'data/station00.csv',
        'data/station04.csv',
        'data/station05.csv',
        'data/station09.csv'
    ]
    
    print_step(1, "检查必要文件")
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"❌ 缺少文件: {file}")
        else:
            print(f"✅ 找到文件: {file}")
    
    if missing_files:
        print(f"\n❌ 缺少 {len(missing_files)} 个必要文件，无法继续分析")
        return False
    
    # 创建结果目录
    print_step(2, "创建结果目录")
    results_dir = Path('results')
    figures_dir = results_dir / 'figures'
    
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    print(f"✅ 结果目录已创建: {results_dir}")
    print(f"✅ 图表目录已创建: {figures_dir}")
    
    # 运行主要分析
    print_step(3, "运行NWP信息融入的预测模型分析")
    print("🔄 正在运行 improved_power_prediction_with_nwp.py...")
    
    try:
        # 导入并运行主分析
        import improved_power_prediction_with_nwp as nwp_analysis
        
        # 运行分析
        start_time = time.time()
        nwp_analysis.run_nwp_analysis()
        end_time = time.time()
        
        print(f"✅ NWP分析完成，耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"❌ NWP分析运行失败: {str(e)}")
        return False
    
    # 运行评估分析
    print_step(4, "运行评估指标分析")
    print("🔄 正在运行 evaluation_metrics_nwp.py...")
    
    try:
        # 导入并运行评估分析
        import evaluation_metrics_nwp as eval_analysis
        
        # 运行评估
        start_time = time.time()
        eval_analysis.evaluate_nwp_predictions()
        end_time = time.time()
        
        print(f"✅ 评估分析完成，耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 评估分析运行失败: {str(e)}")
        return False
    
    # 检查生成的结果
    print_step(5, "检查生成的结果")
    
    # 检查模型文件
    model_files = list(results_dir.glob('*_xgboost_model.pkl'))
    print(f"📊 生成的模型文件: {len(model_files)} 个")
    for model_file in model_files:
        print(f"  ✅ {model_file.name}")
    
    # 检查预测结果文件
    prediction_files = list(results_dir.glob('*_prediction_results.csv'))
    print(f"📈 生成的预测结果文件: {len(prediction_files)} 个")
    for pred_file in prediction_files:
        print(f"  ✅ {pred_file.name}")
    
    # 检查评价指标文件
    metrics_files = list(results_dir.glob('*_metrics.csv'))
    print(f"📋 生成的评价指标文件: {len(metrics_files)} 个")
    for metrics_file in metrics_files:
        print(f"  ✅ {metrics_file.name}")
    
    # 检查图表文件
    figure_files = list(figures_dir.glob('*.png'))
    print(f"📊 生成的图表文件: {len(figure_files)} 个")
    for fig_file in sorted(figure_files):
        print(f"  ✅ {fig_file.name}")
    
    # 生成总结报告
    print_step(6, "生成总结报告")
    
    try:
        # 读取多站点对比结果
        comparison_file = results_dir / 'multi_station_comparison.csv'
        if comparison_file.exists():
            import pandas as pd
            comparison_df = pd.read_csv(comparison_file)
            
            print("\n📊 多站点性能对比:")
            print(comparison_df.to_string(index=False))
            
            # 计算平均性能
            avg_rmse = comparison_df['RMSE'].mean()
            avg_accuracy = comparison_df['准确率(%)'].mean()
            avg_correlation = comparison_df['相关系数'].mean()
            
            print(f"\n📈 平均性能指标:")
            print(f"  平均RMSE: {avg_rmse:.4f}")
            print(f"  平均准确率: {avg_accuracy:.2f}%")
            print(f"  平均相关系数: {avg_correlation:.4f}")
        
    except Exception as e:
        print(f"⚠️ 读取对比结果时出错: {str(e)}")
    
    # 完成总结
    print_header("Problem 3 分析完成")
    
    print("🎉 所有分析已成功完成！")
    print("\n📁 结果文件位置:")
    print(f"  📊 模型文件: {results_dir}/")
    print(f"  📈 图表文件: {figures_dir}/")
    print(f"  📋 报告文件: problem3_summary_report.md")
    
    print("\n🔍 主要成果:")
    print("  ✅ 融入NWP信息的预测模型已训练完成")
    print("  ✅ 多模型对比分析已完成")
    print("  ✅ NWP信息有效性评估已完成")
    print("  ✅ 场景化分析已完成")
    print("  ✅ 综合可视化图表已生成")
    print("  ✅ 多站点性能对比已完成")
    
    print("\n📖 查看详细报告:")
    print("  cat problem3_summary_report.md")
    
    return True

def main():
    """主函数"""
    try:
        success = run_analysis()
        if success:
            print("\n🎯 Problem 3 分析成功完成！")
            sys.exit(0)
        else:
            print("\n❌ Problem 3 分析失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了分析过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 