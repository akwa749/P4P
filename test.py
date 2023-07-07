import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('./data/P001 2022-01-25 01_39_54.ply')
# point_cloud = np.asarray(pcd.points)

obb = pcd.get_oriented_bounding_box()

# Create a line set to represent the edges of the box


# edge_coordinates = [np.asarray(obb.get_box_points())[i] for i in edges]
# lines = o3d.geometry.LineSet()
obb.color = [1,0,0]
# lines.points = obb.get_box_points()

# lines.lines = o3d.utility?.Vector2iVector(edges)

# Visualize
o3d.visualization.draw_geometries([pcd, obb])
