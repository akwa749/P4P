import open3d as o3d
import numpy as np

#testing git

# # Read the point cloud file
# with open("C:/Users/Anthony/Documents/Research Project/testing point clouds/P001 2022-01-25 01_39_54.ply", 'r') as file:
#     pcd = o3d.io.read_point_cloud(file)
#     # Visualize the point cloud
#     o3d.visualization.draw_geometries([pcd])

#     file.close()

pcd = o3d.io.read_point_cloud('P001 2022-01-25 01_39_54.ply')

#getting max and minimum points from the point cloud data

# Get the minimum and maximum values for each dimension
min_point = np.min(np.asarray(pcd.points), axis=0)
max_point = np.max(np.asarray(pcd.points), axis=0)


print(min_point)
print(max_point)

## removing background

points = np.asarray(pcd.points)

##getting box

#z coordinate is our height so want to keep it same as the min and max point
#x provides the width of the human so like one hand to the other


# #floor remover
# floor = 0.08

# roi = o3d.geometry.AxisAlignedBoundingBox(
#     min_bound=(-0.5, -0.3, min_point[2] + floor),
#     max_bound=(0.7, 0.7, max_point[2])
# )

# cropped = pcd.crop(roi)

# ###
# colours = pcd.colors
# print(colours)

o3d.visualization.draw_geometries([pcd])


# Visualize the point cloud
#o3d.visualization.draw_geometries([pcd])