import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import numpy as np

# generate some random points
points = np.random.rand(10, 2)

# generate Voronoi diagram and plot
vor = Voronoi(points)
voronoi_plot_2d(vor)
plt.title('Voronoi Diagram')
plt.show()

# generate Delaunay triangulation and plot
tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.title('Delaunay Triangulation')
plt.show()

