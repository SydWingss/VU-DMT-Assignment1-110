import pandas as pd
import numpy as np
import os
import re

def divide_by_variable(df):
    folder_name = 'divide_by_variable'
# 获取当前工作目录
    current_directory = os.getcwd()
# 构造子文件夹的完整路径
    folder_path = os.path.join(current_directory, folder_name)
# 如果子文件夹不存在，则创建它
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    variable_list = df['variable'].unique()
    for i,variable in enumerate(variable_list):

        #把整张表拆分成多个子表，每个子表只包含一个变量的数据。并把子表存入新文件夹 divide_by_variable 中。
        sub_df = df[df['variable'] == variable]
        sub_df.drop(columns=['variable'], inplace=True)
        sub_df.rename(columns={'value': variable}, inplace=True)    
        # 定义CSV文件的名称
        csv_file_name = variable + '.csv'
        # 构造CSV文件的完整路径
        csv_file_path = os.path.join(folder_path, csv_file_name)
        # 将DataFrame保存到CSV文件中
        sub_df.to_csv(csv_file_path, index=False)


def trans_to_sparse_matrix(df):

    df_pivot = df.pivot_table(index=['id', 'time'], columns='variable', values='value', aggfunc='first').reset_index()
    unique_ids = df['variable'].unique()
    unique_ids_list = list(unique_ids)
    # 确保结果中只包含指定的列
    df_final = df_pivot[['id', 'time'] + unique_ids_list]

    # 定义子文件夹名称
    folder_name = 'sparse_matrix'

    # 获取当前工作目录
    current_directory = os.getcwd()

    # 构造子文件夹的完整路径
    folder_path = os.path.join(current_directory, folder_name)

    # 如果子文件夹不存在，则创建它
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 定义CSV文件的名称
    csv_file_name = 'sparse_matrix_data.csv'

# 构造CSV文件的完整路径
    csv_file_path = os.path.join(folder_path, csv_file_name)

# 将DataFrame保存到CSV文件中
    df_final.to_csv(csv_file_path, index=False)

def divide_by_id(df):
    folder_name = 'id'
    # 获取当前工作目录
    current_directory = os.getcwd()
# 构造子文件夹的完整路径
    folder_path = os.path.join(current_directory, folder_name)
# 如果子文件夹不存在，则创建它
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    id_list = df['id'].unique()
    for i,id in enumerate(id_list):
        #把整张表拆分成多个子表，每个子表只包含一个变量的数据。并把子表存入新文件夹 divide_by_variable 中。
        sub_df = df[df['id'] == id]
   
        # 定义CSV文件的名称
        csv_file_name = id + '.csv'
        # 构造CSV文件的完整路径
        csv_file_path = os.path.join(folder_path, csv_file_name)
        # 将DataFrame保存到CSV文件中
        sub_df.to_csv(csv_file_path, index=False)


def time_resampling(df):
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    result = df.groupby('id').apply(lambda x: x.select_dtypes(include=[np.number]).resample('D').mean())
    result = result.reset_index(drop=False)
    folder_name ="time_resampling"
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    csv_file_name = 'time_resamping_sparse_matrix_data.csv'
    csv_file_path = os.path.join(folder_path, csv_file_name)
    result.to_csv(csv_file_path, index=False)
    
def clean_raw(data, data_schema):
    invalid_rows = []
    error_types = []
    for index, row in data.iterrows():
        # 检查时间是否符合规定
        if not re.match(data_schema["time"]["pattern"], row["time"]):
            invalid_rows.append(index + 1)
            error_types.append("wrong_time")
            continue

        # 检查变量是否符合规定
        variable = row["variable"]
        value = row["value"]
        if variable in data_schema["variable"]:
            var_schema = data_schema["variable"][variable]
            if var_schema["range"] is not None:
                min_value, max_value = var_schema["range"]
                if not min_value <= value <= max_value:
                    if pd.isnull(value):
                        invalid_rows.append(index + 1)
                        error_types.append("missing_value")
                    else:
                        invalid_rows.append(index + 1)
                        error_types.append("out_of_range")
        else:
            invalid_rows.append(index + 1)
            error_types.append("missing_variable")

    print(len(invalid_rows))
    print("invalid row id:", invalid_rows)
    print("error types:", error_types)
    
def fill_default_value_and_add_mood_level(df):
    
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df_filled = pd.DataFrame()
    columns = ['mood']
    for id, group in df.groupby('id'):
        # 为每个id的数据填充缺失值
        group_rolling = group[columns].rolling('5D').mean()
        group_filled = group[columns].where(group[columns].notna(), group_rolling)
        group_filled.fillna(group[columns].mean(), inplace=True)

  
        # 将填充后的数据与原始数据合并
        group = group.drop(columns, axis=1)
        group = pd.concat([group, group_filled], axis=1)
  
        # 重置索引，将'time'列变为普通列
        group.reset_index(inplace=True)
        # 将'id'列添加到组中，并将其设置为索引
        group['id'] = id
        group.set_index('id', inplace=True)
    
        # 将处理过的组添加到结果DataFrame中
        df_filled = pd.concat([df_filled, group])
    
    low_boundary = 6
    high_boundary = 8
    # Create a new column 'mood_level'
    df_filled.loc[df_filled["mood"]<=low_boundary,'mood_level'] = 'low'
    df_filled.loc[(df_filled['mood'] > low_boundary) & (df_filled['mood'] < high_boundary), 'mood_level'] = 'medium'
    df_filled.loc[df_filled['mood'] >= high_boundary, 'mood_level'] = 'high' 
    
    folder_name ="time_resampling"
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    csv_file_name = 'time_resamping_with_moodlevel.csv'
    csv_file_path = os.path.join(folder_path, csv_file_name)
    
            # 获取列的列表
    cols = list(df_filled.columns)
    # 移除'mood'
    cols.remove('mood')
    # 在第2列的位置插入'mood'
    cols.insert(1, 'mood')
    # 重新索引DataFrame
    df_filled = df_filled.reindex(columns=cols)
    
    df_filled.reset_index().to_csv(csv_file_path, index=False)
    
    