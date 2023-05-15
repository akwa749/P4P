import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


pcd = o3d.io.read_point_cloud('P001 2022-01-25 01_39_54.ply')
point_cloud = np.asarray(pcd.points)



#####################

## this was radius outlier didnt work

# # Compute the centroid of the point cloud
# centroid = np.mean(point_cloud, axis=0)

# # Compute the distances from the centroid to all points
# distances = np.linalg.norm(point_cloud - centroid, axis=1)

# # Define the radius
# radius = np.percentile(distances, 97)  # for example, you could use the 95th percentile distance

# # Keep only the points within the radius
# human_points = point_cloud[distances <= radius]

############################
## statisticla outlier removal

# from sklearn.neighbors import LocalOutlierFactor

# # Initialize the LocalOutlierFactor model
# lof = LocalOutlierFactor(n_neighbors=20, contamination=0.4)

# # Fit the model and predict the outliers
# labels = lof.fit_predict(point_cloud)

# # Keep only the inliers (label == 1)
# human_points = point_cloud[labels == 1]


# human_points = point_cloud[labels == 0]
# background_points = point_cloud[labels == -1]

##############

# from sklearn.ensemble import IsolationForest

# # Initialize the IsolationForest model
# iso = IsolationForest(contamination=0.05)

# # Fit the model and predict the outliers
# labels = iso.fit_predict(point_cloud)

# # Keep only the inliers (label == 1)
# human_points = point_cloud[labels == 1]

############

#bounding box approach

# Define the bounding box
min_x = np.min(point_cloud[:, 0])
max_x = np.max(point_cloud[:, 0])
min_y = np.min(point_cloud[:, 1])
max_y = np.max(point_cloud[:, 1])
min_z = np.min(point_cloud[:, 2])
max_z = np.max(point_cloud[:, 2])

print(min_x)

# Expand the bounding box slightly
padding = 0.5  # for example, 10% of the range

min_x = padding * min_x
max_x = padding * max_x
min_y = padding * min_y
max_y = padding * max_y
min_z = padding * min_z
#z neutral as don't want to cut off head
max_z = 1 * max_z

print(min_x)

# Filter the points
human_points = point_cloud[
    (point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) &
    (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y) &
    (point_cloud[:, 2] >= min_z) & (point_cloud[:, 2] <= max_z)
]


###
visualized_point_cloud = o3d.geometry.PointCloud()
visualized_point_cloud.points = o3d.utility.Vector3dVector(human_points)
o3d.visualization.draw_geometries([visualized_point_cloud])

print("done")