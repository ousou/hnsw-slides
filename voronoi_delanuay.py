import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import numpy as np
import heapq
from copy import deepcopy


np.random.seed(7)
point_count = 256
# generate some random points
points = np.random.rand(point_count, 2)

# generate Voronoi diagram and plot
#vor = Voronoi(points)
#voronoi_plot_2d(vor)
#plt.title('Voronoi Diagram')
#plt.show()

# generate Delaunay triangulation and plot
tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.axis('off')


# create graph
graph = {}
for i in range(points.shape[0]):
    graph[i] = set(tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]])


# greedy search algorithm
def greedy_search(graph, points, start, end):
    heap = [(np.linalg.norm(points[end] - points[start]), start, [start])]
    while heap:
        (cost, current_node, path) = heapq.heappop(heap)
        if current_node == end:
            return path
        for neighbor in graph[current_node]:
            if neighbor not in path:
                total_cost = cost + np.linalg.norm(points[neighbor] - points[end])
                heapq.heappush(heap, (total_cost, neighbor, path + [neighbor]))

# find pair of points for which greedy search algorithm returns longest path
pairs_to_try = 50
random_pairs = np.random.randint(0, len(points), (pairs_to_try, 2))
longest_path = []
longest_path_start = -1
longest_path_end = -1
for i,j in random_pairs:
    path = greedy_search(graph, points, i, j)
    if len(path) > len(longest_path):
        longest_path = path
        longest_path_start = i
        longest_path_end = j

plt.savefig("delaunay_triangulation_256_points.svg")
# plot longest path
longest_path_points = np.array([points[i] for i in longest_path])
longest_path_end_points = np.array([points[longest_path_start], points[longest_path_end]])
print(f'Longest path length: {len(longest_path)}')
plt.plot(longest_path_end_points[:,0], longest_path_end_points[:,1], 'o', color='red')
plt.savefig("delaunay_triangulation_start_end_points.svg")
plt.plot(longest_path_points[:, 0], longest_path_points[:, 1], 'r-', linewidth=4.0)
plt.savefig("delaunay_triangulation_longest_path.svg")
#plt.show()

random_edge_count = 16
random_edges = np.random.randint(0, len(points), (random_edge_count, 2))
random_edge_points = points[random_edges]
#plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
#plt.plot(points[:,0], points[:,1], 'o')
plt.plot(random_edge_points.T[0], random_edge_points.T[1], 'y-')
plt.savefig("delaunay_triangulation_random_edges.svg")

graph_with_random_edges = deepcopy(graph)

for i,j in random_edges:
    graph_with_random_edges[i].add(j)
    graph_with_random_edges[j].add(i)

new_path = greedy_search(graph_with_random_edges, points, longest_path_start, longest_path_end)
new_path_points = np.array([points[i] for i in new_path])
#plt.plot(longest_path_points[:, 0], longest_path_points[:, 1], 'r-', linewidth=4.0)
#plt.plot(longest_path_end_points[:,0], longest_path_end_points[:,1], 'o', color='red')

plt.plot(new_path_points[:, 0], new_path_points[:, 1], 'k-', linewidth=4.0)
plt.savefig("delaunay_triangulation_random_edges_shortest_path.svg")
print(f'New Longest path length: {len(new_path)}')

#plt.title(f'Delaunay Triangulation and random edges, path length {len(new_path)}')

#plt.show()
