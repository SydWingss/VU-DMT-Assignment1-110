import pandas as pd
import numpy as np
import os

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