import functools
import numpy as np

class Graph:
    def __init__(self, N, edges_lists):
        self.vertices_count = N
        self.edges_lists = edges_lists
        self.edges_count = functools.reduce(lambda x, y: x + y, [len(l) for l in edges_lists]) // 2

    def get_degree_stats(self):
        maximum_number_of_edges = self.vertices_count * (self.vertices_count - 1)
        dks = {}
        esp = {}

        for es in self.edges_lists:
            if len(es) not in dks:
                dks[len(es)] = 1 / self.vertices_count
            else:
                dks[len(es)] += 1 / self.vertices_count

        for i in range(self.vertices_count):
            for j in range(i + 1, self.vertices_count):
                if j not in self.edges_lists[i]:
                    continue

                k = len(self.edges_lists[i].intersection(self.edges_lists[j]))

                if k not in esp:
                    esp[k] = 1 / self.edges_count
                else:
                    esp[k] += 1 / self.edges_count

        dist_mat = self.vertices_count + np.zeros((self.vertices_count, self.vertices_count), dtype = 'int64')
        for i in range(self.vertices_count):
            dist_mat[i][list(self.edges_lists[i])] = 1

        for i in range(self.vertices_count):
            for j in range(i + 1, self.vertices_count):
                for k in range(self.vertices_count):
                    if dist_mat[i][k] < self.vertices_count and dist_mat[k][j] < self.vertices_count:
                        dist_mat[i][j] = min(dist_mat[i][j], dist_mat[i][k] + dist_mat[k][j])

        dist_mat = np.triu(dist_mat)
        geodesic_counts = np.bincount(dist_mat.reshape(-1))
        geodesic_counts[0] = geodesic_counts[self.vertices_count] - self.vertices_count

        return dks, esp, geodesic_counts[:-1] / (self.vertices_count * (self.vertices_count - 1) / 2)
