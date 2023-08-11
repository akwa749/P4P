import open3d as o3d
import numpy as np 

pcd = o3d.io.read_point_cloud('C:/Users/Anthony/Documents/2023Sem1/Research Project/Alex_crop.ply')
#o3d.visualization.draw_geometries([pcd])


#################################
# Uniform down-sampling:
#downsampling increases holes but can star model fit against data with holes

# uniform_pcd = pcd.uniform_down_sample(every_k_points=200)
# o3d.visualization.draw_geometries([uniform_pcd])

######################################

#radius outlier removal filter
#computationally expensive takes ages to run

# pcd_rad, ind_rad = pcd.remove_radius_outlier(nb_points=50, radius=0.1)
# outlier_rad_pcd = pcd.select_by_index(ind_rad, invert=True)
# outlier_rad_pcd.paint_uniform_color([1., 0., 1.])
# o3d.visualization.draw_geometries([outlier_rad_pcd])

##################################

#voxel downsampling
voxel_pcd = pcd.voxel_down_sample(voxel_size = 0.025)
o3d.visualization.draw_geometries([voxel_pcd])