import pandas as pd

# 查看station00数据结构
print("=== Station00 数据结构 ===")
df = pd.read_csv('data/station00.csv')
print("列名:", df.columns.tolist())
print("数据形状:", df.shape)
print("前3行数据:")
print(df.head(3))
print("\n数据类型:")
print(df.dtypes)

print("\n=== NWP字段统计 ===")
nwp_cols = [col for col in df.columns if 'nwp_' in col]
print("NWP字段:", nwp_cols)

print("\n=== LMD字段统计 ===")
lmd_cols = [col for col in df.columns if 'lmd_' in col]
print("LMD字段:", lmd_cols)

print("\n=== 数据范围统计 ===")
print(df.describe()) 