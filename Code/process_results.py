import pickle
import numpy as np
import pandas as pd

f = open('student_t_m1.txt', 'rb')

m = pickle.load(f)
"""
params = pickle.load(f)
params = [(params[0], params[1])]
print(params)
"""
params = np.asarray(pickle.load(f))
sigmas = np.asarray(pickle.load(f))
original_edges_counts = np.asarray(pickle.load(f))
found_edges_counts = np.asarray(pickle.load(f))
total_variance_graph_statistics = np.asarray(pickle.load(f))

df = pd.DataFrame({'Params': params})
df['Original Edges'] = original_edges_counts.mean(axis = 1)
df['Original Edges Std'] = original_edges_counts.std(axis = 1)
df['Found Edges'] = found_edges_counts.mean(axis = 1)
df['Found Edges Std'] = found_edges_counts.std(axis = 1)

total_variance_graph_statistics = np.asarray(total_variance_graph_statistics)

df['Degree Mean'] = total_variance_graph_statistics.mean(axis = 2).T[0]
df['Degree Std'] = total_variance_graph_statistics.std(axis = 2).T[0]
df['ESP Mean'] = total_variance_graph_statistics.mean(axis = 2).T[1]
df['ESP Std'] = total_variance_graph_statistics.std(axis = 2).T[1]
df['Geodesic Mean'] = total_variance_graph_statistics.mean(axis = 2).T[2]
df['Geodesic Std'] = total_variance_graph_statistics.std(axis = 2).T[2]
df['Found Sigma'] = sigmas.mean(axis = 1)

print(df)
#del df['Params']
df.to_csv('student_t_m1.csv', index = False)
