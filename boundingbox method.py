import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('./data/P002 2022-01-25 01_24_48.ply')
point_cloud = np.asarray(pcd.points)

def bounding_box():
    # Define the bounding box
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Expand the bounding box slightly
    padding = 0.5  # for example, 10% of the range

    min_x = padding * min_x
    max_x = padding * max_x
    min_y = padding * min_y
    max_y = padding * max_y
    min_z = padding * min_z

    # Filter the points
    human_points = point_cloud[
        (point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) &
        (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y) &
        (point_cloud[:, 2] >= min_z) & (point_cloud[:, 2] <= max_z)
    ]


    ###
    visualized_point_cloud = o3d.geometry.PointCloud()
    visualized_point_cloud.points = o3d.utility.Vector3dVector(human_points)
    obb = visualized_point_cloud.get_minimal_oriented_bounding_box()
    obb.color = [1,0,0]
    # abb = visualized_point_cloud.get_oriented_bounding_box()
    # abb.color = [0,1,0]
    o3d.visualization.draw_geometries([obb,visualized_point_cloud])
    

    print("done")

bounding_box()