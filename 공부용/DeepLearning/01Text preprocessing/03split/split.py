import pandas as pd 
values = [['data1', 1],
['data2', 0],
['data3', 0],
['data4', 1]]
columns = ['데이터', '분류']

df = pd.DataFrame(values, columns=columns)
print(df)