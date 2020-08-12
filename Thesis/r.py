import pandas as pd

df = pd.read_csv('student_t_m1.csv')
df = df.round(3)
print(df)

df['Original Edges'] = df['Original Edges'].astype(str) + ' ' + u"\u00B1" + ' ' + df['Original Edges Std'].astype(str)
df['Found Edges'] = df['Found Edges'].astype(str) + ' ' + u"\u00B1" + ' ' + df['Found Edges Std'].astype(str)
df['Degree Mean'] = df['Degree Mean'].astype(str) + ' ' + u"\u00B1" + ' ' + df['Degree Std'].astype(str)
df['ESP Mean'] = df['ESP Mean'].astype(str) + ' ' + u"\u00B1" + ' ' + df['ESP Std'].astype(str)
df['Geodesic Mean'] = df['Geodesic Mean'].astype(str) + ' ' + u"\u00B1" + ' ' + df['Geodesic Std'].astype(str)

del df['Geodesic Std']
del df['ESP Std']
del df['Original Edges Std']
del df['Found Edges Std']
del df['Degree Std']

df.to_csv('student_t_m1.csv', index = False)
