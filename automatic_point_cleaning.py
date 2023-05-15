import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


pcd = o3d.io.read_point_cloud('./data/P001 2022-01-25 01_39_54.ply')
points = np.asarray(pcd.points)

#getting max and minimum points from the point cloud data

# Get the minimum and maximum values for each dimension
min_point = np.min(np.asarray(pcd.points), axis=0)
max_point = np.max(np.asarray(pcd.points), axis=0)

#getting the centroid of the model
centroid = np.mean(pcd.points, axis=0)

dist_from_centroid = np.linalg.norm(points - centroid, axis=1)
X_with_features = np.hstack((points, dist_from_centroid[:, np.newaxis]))


#radius around centroid (search radius)
#will need to automate this later on
radius = 1

#one cluster for human one for background
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_with_features)

y_pred = kmeans.predict(X_with_features)

# Get the cluster centers and assign the human model to the cluster with the higher z-coordinate center
centers = kmeans.cluster_centers_
human_cluster = np.argmax(centers[:, 2])
y = (y_pred == human_cluster).astype(int)


##random forest classifcation
X_train, X_test, y_train, y_test = train_test_split(X_with_features, y, test_size=0.2, random_state=42)

# Create a random forest classifier with 100 trees
rfc = RandomForestClassifier(n_estimators=1000)

# Fit the random forest classifier to the training data
rfc.fit(X_with_features, y_train)

dist_from_centroid_all = np.linalg.norm(points - centroid, axis=1)
X_with_features_all = np.hstack((points, dist_from_centroid_all[:, np.newaxis]))


# Make predictions on the test data
y_pred = rfc.predict(X_with_features_all)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

##mask
mask = (y_pred == 0)
# Create a new point cloud with only the points corresponding to the human model
pcd_human = pcd.select_by_index(np.where(mask)[0])

# Visualize the human model
o3d.visualization.draw_geometries([pcd])


