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

edge_count = sum(len(graph[i]) for i in graph) / 2
print("edge_count", edge_count)

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

def greedy_search_recursive(graph, points, start, end):
    return _greedy_search_2(graph, points, [start], end)

def _greedy_search_2(graph, points, current_path, end):
    if current_path[-1] == end:
        return current_path
    closest_neighbor = None
    for neighbor in graph[current_path[-1]]:
        if neighbor not in current_path:
            if closest_neighbor is None:
                closest_neighbor = neighbor
            elif np.linalg.norm(points[neighbor] - points[end]) < np.linalg.norm(points[closest_neighbor] - points[end]):
                closest_neighbor = neighbor
    current_path.append(closest_neighbor)
    return _greedy_search_2(graph, points, current_path, end)

# find pair of points for which greedy search algorithm returns longest path
pairs_to_try = 50
random_pairs = np.random.randint(0, len(points), (pairs_to_try, 2))
longest_path = []
longest_path_start = -1
longest_path_end = -1
for i,j in random_pairs:
    path = greedy_search_recursive(graph, points, i, j)
    if len(path) > len(longest_path):
        longest_path = path
        longest_path_start = i
        longest_path_end = j

print("longest_path_start", longest_path_start)
print("longest_path_end", longest_path_end)
print("longest_path", longest_path)

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

random_edge_count = 32
random_edges = np.random.randint(0, len(points), (random_edge_count, 2))
random_edge_points = points[random_edges]
#plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
#plt.plot(points[:,0], points[:,1], 'o')
plt.plot(random_edge_points.T[0], random_edge_points.T[1], 'y-')
plt.savefig("delaunay_triangulation_random_edges.svg")

graph_with_random_edges = deepcopy(graph)
print("random_edges", random_edges)

for i,j in random_edges:
    graph_with_random_edges[i].add(j)
    graph_with_random_edges[j].add(i)


new_path = greedy_search_recursive(graph_with_random_edges, points, longest_path_start, longest_path_end)
print("new_path", new_path)
new_path_points = np.array([points[i] for i in new_path])
#plt.plot(longest_path_points[:, 0], longest_path_points[:, 1], 'r-', linewidth=4.0)
#plt.plot(longest_path_end_points[:,0], longest_path_end_points[:,1], 'o', color='red')

plt.plot(new_path_points[:, 0], new_path_points[:, 1], 'k-', linewidth=4.0)
plt.savefig("delaunay_triangulation_random_edges_shortest_path.svg")
print(f'New Longest path length: {len(new_path)}')

#plt.title(f'Delaunay Triangulation and random edges, path length {len(new_path)}')

#plt.show()
