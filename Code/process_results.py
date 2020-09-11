import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'test_skewed_t_m1'
f = open(filename + '.txt', 'rb')

m = pickle.load(f)
"""
params = pickle.load(f)
params = [(params[0], params[1])]
print(params)
"""
#pickle.load(f)
params = np.asarray(pickle.load(f))

sigmas = np.asarray(pickle.load(f))
original_edges_counts = np.asarray(pickle.load(f))
found_edges_counts = np.asarray(pickle.load(f))
total_variance_graph_statistics = np.asarray(pickle.load(f))

df = pd.DataFrame()
#df['Params'] = params
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

df = df.round(3)

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

#del df['Params']
df.to_csv(filename + '.csv', index = False)
