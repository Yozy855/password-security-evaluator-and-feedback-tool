import pandas as pd

df = pd.read_csv("data.csv", on_bad_lines = 'skip')
#print last 5 rows
#print 5 random rows
print(df.sample(5))
print(df.tail())
