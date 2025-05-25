import pandas as pd
from datetime import datetime, timedelta
import os # 引入os模块用于检查文件和目录

def transform_csv(input_file, output_file_template, station_name):
    # 1. 读取CSV文件，自动推断表头
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：输入文件 {input_file} 未找到。")
        return
    except pd.errors.EmptyDataError:
        print(f"错误：输入文件 {input_file} 为空。")
        return
    except Exception as e:
        print(f"读取CSV文件 {input_file} 时发生错误: {e}")
        return

    # 2. 检查原始列名是否符合预期 (你的数据示例中的表头)
    expected_original_cols = ['date_time', 'actual_power', 'predicted_power', 'error']
    if not all(col in df.columns for col in expected_original_cols):
        print(f"警告：文件 {input_file} 的列名与预期 ({expected_original_cols}) 不完全相符。")
        print(f"文件实际列名: {df.columns.tolist()}")
        # 如果列的数量正确，但名称不完全匹配，可以尝试强制重命名
        if len(df.columns) == len(expected_original_cols):
            print(f"由于列数匹配，将尝试使用预期的列名: {expected_original_cols}")
            df.columns = expected_original_cols
        else:
            print(f"错误：列数 ({len(df.columns)}) 与预期 ({len(expected_original_cols)}) 不匹配。请检查CSV文件格式。")
            return

    # 3. 转换 'date_time' 列并创建新的时间列
    try:
        df['date_time'] = pd.to_datetime(df['date_time'])
    except Exception as e:
        print(f"错误：转换 'date_time' 列为日期时间对象时失败: {e}")
        return

    # 创建 '起报时间' 和 '预报时间' 列
    df['起报时间'] = df['date_time'].dt.floor('D').dt.strftime('%Y/%m/%d/00:00')
    df['预报时间'] = df['date_time'].dt.strftime('%Y/%m/%d/%H:%M')

    # 4. 重命名其他列以符合输出要求
    df = df.rename(columns={
        'actual_power': '实际功率(MW)',
        'predicted_power': 'XGBoost 预测功率(MW)', # 你可以根据需要修改模型名称
        'error': '误差'
    })

    # 5. 定义并检查输出列的顺序
    # 注意：现在使用中文列名 '起报时间' 和 '预报时间'
    output_columns = ['起报时间', '预报时间', '实际功率(MW)', 'XGBoost 预测功率(MW)', '误差']

    missing_output_cols = [col for col in output_columns if col not in df.columns]
    if missing_output_cols:
        print(f"错误：以下目标输出列在DataFrame中缺失: {missing_output_cols}")
        print(f"DataFrame当前可用列: {df.columns.tolist()}")
        return

    df_output = df[output_columns]

    # 构建实际的输出文件名
    # 假设 "第6月 7 天" 是固定的，你可以根据需要修改这部分
    # 为了演示，我将使用 "第X月Y天" 作为占位符，实际应用中你可能需要从数据中提取月份或有其他逻辑
    # 这里简化为固定的后缀文本
    output_filename_actual = output_file_template.replace("{station_name}", station_name)
    
    # 6. 保存到新的CSV文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_filename_actual)
        if output_dir and not os.path.exists(output_dir): # output_dir可能为空（如果在当前目录）
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        df_output.to_csv(output_filename_actual, index=False, encoding='utf-8-sig')
        print(f"转换完成。输出已保存到 {output_filename_actual}") # 在函数内部打印成功信息
    except Exception as e:
        print(f"保存到CSV文件 {output_filename_actual} 时发生错误: {e}")
        return

def process_all_stations():
    stations = ['00', '04', '05', '09']
    base_input_dir = 'results' # 定义基础输入目录
    # 定义输出文件名的模板，包含站点名和固定的描述部分
    # 你可以修改 "第X月Y天" 这部分以适应你的实际需求
    output_file_suffix = "_第6月7天的功率预测结果.csv" # 修改为你需要的固定后缀

    for station_code in stations:
        station_name_for_file = f"station{station_code}" # 用于文件名的站点名部分
        input_file = os.path.join(base_input_dir, f'{station_name_for_file}_prediction_results.csv')
        
        # 使用站点名和固定的后缀构建输出文件名
        output_file_name_template_actual = os.path.join(base_input_dir, f'{station_name_for_file}{output_file_suffix}')

        print(f"Processing station {station_code}...")
        print(f"  Input file: {input_file}")
        print(f"  Output file will be: {output_file_name_template_actual.replace('{station_name}', station_name_for_file)}") # 预览将要生成的文件名

        if not os.path.exists(input_file):
            print(f"警告：输入文件 {input_file} 不存在，跳过此站点。")
            print("-" * 30)
            continue # 跳到下一个站点

        transform_csv(input_file, output_file_name_template_actual, station_name_for_file)
        # 注意：成功的打印移到了transform_csv函数内部
        print("-" * 30)

if __name__ == '__main__':
    results_dir_main = 'results'
    if not os.path.exists(results_dir_main):
        os.makedirs(results_dir_main)
        print(f"创建目录: {results_dir_main}")
        # 为了测试，可以创建一些虚拟的 stationXX_prediction_results.csv 文件
        # for s_demo_code in ['00', '04', '05', '09']:
        #     s_demo_name = f'station{s_demo_code}'
        #     demo_input_file = os.path.join(results_dir_main, f'{s_demo_name}_prediction_results.csv')
        #     if not os.path.exists(demo_input_file):
        #         dummy_data = {
        #             'date_time': ['2019-06-06 16:00:00', '2019-06-06 16:15:00', '2024-06-01 10:00:00'],
        #             'actual_power': [0.0, 0.0, 50.5],
        #             'predicted_power': [0.0, 0.0, 48.2],
        #             'error': [0.0, 0.0, 2.3]
        #         }
        #         pd.DataFrame(dummy_data).to_csv(demo_input_file, index=False)
        #         print(f"创建虚拟测试文件: {demo_input_file}")

    process_all_stations()