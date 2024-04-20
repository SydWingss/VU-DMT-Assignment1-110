import pandas as pd

# 创建一个包含很长的单元格的数据表
data = {
    'Name': ['Alex', 'Bob', 'Cathy', 'David', 'Eva'],
    'Comment': ['This is a very very very very very very very very very very very very very very very very very very very very very long comment', 
                'Short comment',
                'This is also a long long long long long long long long long long long long long long long long long long long long long comment', 
                'Short short short', 
                'Another long long long long long long long long long long long long long long long long long long long long long comment']
}
df = pd.DataFrame(data)

# 默认情况下，pandas 会自动调整列的宽度
print(df)

# 设置列宽度的最小值
pd.set_option('display.max_colwidth', None)
print(df)