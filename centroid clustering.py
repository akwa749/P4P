import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud('P001 2022-01-25 01_39_54.ply')
point_cloud = np.asarray(pcd.points)

# Compute the centroid of the point cloud data (assuming the human model is near the center of the point cloud)
centroid = np.mean(point_cloud, axis=0)

# Cluster the points into two groups using k-means clustering
kmeans = KMeans(n_clusters=3, init=np.array([centroid, 0.7*np.min(point_cloud, axis=0), 0.7*np.max(point_cloud, axis = 0)]), n_init=3)
labels = kmeans.fit_predict(point_cloud)

# Separate the points corresponding to the human model and the background based on the cluster labels
human_points = point_cloud[labels == 2]
background_points = point_cloud[labels == 1]

print(pcd)
print(human_points)


visualized_point_cloud = o3d.geometry.PointCloud()
visualized_point_cloud.points = o3d.utility.Vector3dVector(human_points)
o3d.visualization.draw_geometries([visualized_point_cloud])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(human_points[:,0], human_points[:,1], human_points[:,2])

# # Set the axis labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Scatter Plot of Filtered Point Cloud Data')
# plt.show()

# present = o3d.geometry.human_points(oints=o3d.utility.Vector3dVector(point_cloud))
# o3d.visualization.draw_geometries([present])

# print(human_points)