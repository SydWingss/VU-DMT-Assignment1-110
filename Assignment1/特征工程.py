import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
df = pd.read_csv('Assignment1/time_resampling/time_resamping_sparse_matrix_data.csv')

# 将'time'列转换为datetime类型
df['time'] = pd.to_datetime(df['time'])

# 将'time'列设置为索引
df.set_index('time', inplace=True)

# 创建一个新的DataFrame来保存结果
df_filled = pd.DataFrame()

# 定义要进行滚动窗口计算的列
columns = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
onehot_culumns = ['call', 'sms']

# 创建一个MinMaxScaler对象
scaler = MinMaxScaler()

# 对每个唯一的id进行分组
for id, group in df.groupby('id'):
    # 对指定的列进行滚动窗口计算（例如，过去5天）
    group_rolling = group[columns].rolling('5D').mean()

    # 使用滚动窗口计算的结果填充NaN值
    group_filled = group[columns].where(group[columns].notna(), group_rolling)

    # 如果还有NaN值，使用该id的所有数据的平均值来填充
    group_filled.fillna(group[columns].mean(), inplace=True)

    # 对'call'和'sms'列中的NaN值填充为0
    group_filled[onehot_culumns] = group[onehot_culumns]
    group_filled[onehot_culumns] = group_filled[onehot_culumns].fillna(0)

    # 对'call'和'sms'列进行独热编码
    group_filled = pd.get_dummies(group_filled, columns=onehot_culumns)

    # 对指定的列进行归一化
    group_filled[columns] = scaler.fit_transform(group_filled[columns])

    # 重置索引，将'time'列变为普通列
    group_filled.reset_index(inplace=True)
    
    # 将'id'列添加到组中，并将其设置为索引
    group_filled['id'] = id
    group_filled.set_index('id', inplace=True)
    
    # 将处理过的组添加到结果DataFrame中
    df_filled = pd.concat([df_filled, group_filled])

# 保存结果到新的CSV文件
print(df_filled.head())
df_filled.to_csv('Assignment1/time_resampling/filled_time_resamping_sparse_matrix_data.csv')