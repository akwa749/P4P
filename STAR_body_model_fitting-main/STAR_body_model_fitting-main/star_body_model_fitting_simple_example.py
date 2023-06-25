"""
star_body_model_fitting_simple_example.py
by Conrad Werkhoven, Auckland Bioengineering Institute

https://github.com/ConradW01/STAR_body_model_fitting

A simple example of using the STAR body model optimiser to fit a STAR body model to a 3D scanner scan

"""
from pathlib import Path

import open3d as o3d

from star_body_model_optimiser import StarBodyModelOptimiser
from star_body_model_utils import o3d_color_light_red, \
    load_trimesh, ScanType, save_trimesh, \
    get_shape_pc1_and_initial_translation, compute_distance_metrics, calc_cloud_compare_c2c_dist, \
    save_cloud_compare_sampled_point_cloud, td_scan_to_star_transform_matrix
from star_body_model_initial_pose_and_constraints import initial_pose

# Type of scan, either td (3d), lidar, zozo or star. This determines the rotation transformation
# that is applied when loaded and saved
scan_type = ScanType.td
gender = 'male'
# If we sample a mesh we will use 200000 points
n_points_in_pointcloud = 200000

# Number of iterations for optimisation/fitting
max_iters = 1001

# Change this to reflect where the sample scans are

root_folder = 'D:/STAR_body_model_fitting_data'

##changing it for anthony,
root_folder = 'C:/Users/Anthony/Documents/2023Sem1/Research Project/testing point clouds/P4P/Star_body_model_fitting_sample_data-main/Star_body_model_fitting_sample_data-main'


# Name of the target scan we are trying to fit with a STAR body model
# The target point cloud can be created in cloudcompare or similar. We create it by sampling the target scan 200000x
# It is used for calculating fitting metrics
# Input Data
target_scan_filename_full = Path(root_folder, 'Sample_Data/3D scanner - Ground Truth', 'P001 2022-02-07 01_26_37.obj')
target_pointcloud_filename_full = Path(root_folder, 'Sample_Data/3D scanner - Ground Truth',
                                       'P001 2022-02-07 01_26_37_sampled_pointcloud.ply')
target_scan_name = Path(target_scan_filename_full).stem

# Output folder for results
output_folder = Path(root_folder, 'Sample_Output/Simple_Example')

# You will need to install cloudcompare and include path here
cloudcomparepath = Path("C:/Program Files/CloudCompare/CloudCompare")
# We use the same output_folder for cloudcompare
cloudcomparetmpfolder = output_folder

# Make the output folder if it does not exist
if not output_folder.is_dir():
    output_folder.mkdir(parents=True)

# Load mesh or point_cloud target scan
target_scan_trimesh = load_trimesh(target_scan_filename_full, scan_type)

# Create an open3d mesh for drawing to screen
target_scan_o3d = target_scan_trimesh.as_open3d
target_scan_o3d.paint_uniform_color(o3d_color_light_red)

# Calculate the shape pc1 and the initial translation to shift the star model to match the target scan
beta, star_initial_transl = get_shape_pc1_and_initial_translation(gender, target_scan_trimesh)

# Get the optimisation parameters and the upper and lower constraints
combined_parameters, combined_parameters_lower_limit, combined_parameters_upper_limit \
    = initial_pose(star_initial_transl, palm_orientation='front', run_number=1, scan_type=scan_type, beta=beta)

# Create an STAR body model optimiser
opto = StarBodyModelOptimiser(target_scan_trimesh, target_scan_name, scan_type, gender,
                              combined_parameters, combined_parameters_lower_limit, combined_parameters_upper_limit,
                              output_folder, max_iters)

# Run fitting
star_mesh_out_trimesh, star_mesh_out_zero_pose_trimesh = opto.run()

# Filenames for saving - save STAR mesh and the zero pose
star_mesh_out_filename = Path(output_folder, f'{target_scan_name}_{max_iters:04}.obj')
save_trimesh(star_mesh_out_trimesh, scan_type, star_mesh_out_filename)

star_mesh_out_zero_pose_filename = Path(output_folder, f'{target_scan_name}_{max_iters:04}_zero_pose.obj')
save_trimesh(star_mesh_out_zero_pose_trimesh, ScanType.star, star_mesh_out_zero_pose_filename)

# Calculate metrics
print('Now calculating surf 2 surf metrics:')
metrics = compute_distance_metrics(target_scan_filename_full,
                                   star_mesh_out_filename,
                                   sample_size_reference=n_points_in_pointcloud,
                                   sample_size_compared=n_points_in_pointcloud)

print(metrics)

# Calculate the cloudcompare C2C_Distance surface mesh

# First we need to save a sampled pointcloud of the fitted STAR body model to be used by calc_cloud_compare_c2c_dist
save_cloud_compare_sampled_point_cloud(star_mesh_out_filename)

# Now get the STAR sampled pointcloud filename
star_mesh_out_sampled_point_cloud_filename = Path(output_folder,
                                                  f'{target_scan_name}_{max_iters:04}_SAMPLED_POINTS.bin')

# Load in the STAR sampled pointcloud and the target sampled pointcloud and output a cloud compare C2C_Dist surfacemesh
mean, stdev, m_s_string = calc_cloud_compare_c2c_dist(star_mesh_out_sampled_point_cloud_filename,
                                                      target_pointcloud_filename_full, cloudcomparepath,
                                                      cloudcomparetmpfolder)

# Load the C2C_Dist mesh and display it on screen
star_mesh_out_c2c_dist_filename = Path(output_folder, f'{target_scan_name}_{max_iters:04}_C2C_DIST_SURFACE.ply')
mesh_c2c_dist_o3d = o3d.io.read_triangle_mesh(str(star_mesh_out_c2c_dist_filename))
mesh_c2c_dist_o3d = mesh_c2c_dist_o3d.transform(td_scan_to_star_transform_matrix)
mesh_c2c_dist_o3d.compute_vertex_normals()

# Visualise
o3d.visualization.draw_geometries([mesh_c2c_dist_o3d])

print('Finished!')
