"""
star_body_model_fitting_AUT.py
by Conrad Werkhoven, Auckland Bioengineering Institute

https://github.com/ConradW01/STAR_body_model_fitting

Fits the STAR body model to scans from a 3D scanner, a lidar and a zozo suit.
Compares the fitted zero-pose STAR model from the lidar and zozo suit to the zero-pose 'ground-truth' 3D scan

"""
from pathlib import Path
from time import time, ctime
from typing import Tuple

import configobj
import numpy as np
import pandas as pd

from star_body_model_optimiser import StarBodyModelOptimiser
from star_body_model_utils import load_trimesh, ScanType, calc_cloud_compare_c2m_dist, \
    calc_cloud_compare_c2c_dist, save_cloud_compare_sampled_point_cloud, append_string_to_filename, surf2surf_dist, \
    compute_distance_metrics, save_trimesh, \
    get_shape_pc1_and_initial_translation, print_out_torch_info, \
    save_star_mesh_out_and_target_scan_with_no_hand_head_feet
from star_body_model_initial_pose_and_constraints import initial_pose


def main():
    # print out some info regarding torch environment
    print_out_torch_info()

    start_time = time()

    # pull in parameters from the config file
    root_folder, data_subfolder, output_subfolder, td_scan_subfolder, lidar_subfolder, zozo_subfolder, \
        star_fitting_results_subfolder, max_iters, \
        n_points_in_pointcloud, cloud_compare_path, cloud_compare_tmp_folder = \
        read_config_file(Path("star_body_model_fitting_AUT_sample_data.ini"))
        #read_config_file('C:/Users/Anthony/Documents/2023Sem1/Research Project/testing point clouds/P4P/STAR_body_model_fitting-main/STAR_body_model_fitting-main/star_body_model_fitting_AUT_sample_data.ini')
    
    #changing the line above before it was Path('star_body_model_fitting_AUT_sample_data.ini')

    td_scan_data_folder = Path(root_folder, data_subfolder, td_scan_subfolder)  # folder for the 3D scanner input
    lidar_data_folder = Path(root_folder, data_subfolder, lidar_subfolder)  # folder for the lidar input
    zozo_data_folder = Path(root_folder, data_subfolder, zozo_subfolder)  # folder for the zozo input

    td_scan_output_folder = Path(root_folder, output_subfolder, star_fitting_results_subfolder,
                                 td_scan_subfolder)  # folder for the 3D scanner output
    lidar_output_folder = Path(root_folder, output_subfolder, star_fitting_results_subfolder,
                               lidar_subfolder)  # folder for the lidar output
    zozo_output_folder = Path(root_folder, output_subfolder, star_fitting_results_subfolder,
                              zozo_subfolder)  # folder for the zozo output

    output_root_folder = Path(root_folder, output_subfolder)

    # Create some arrays to store results - these will be saved to Excel spreadsheets at the end of run
    results_c2c, results_c2m, results_gias, results_s2s_fitting_metrics_array, results_s2s_zero_pose_metrics_array = \
        create_results_arrays()

    # Keep track of row in results arrays
    s2s_fitting_count = 1
    s2s_zero_pose_count = 1
    s2s_fitting_count_old = 1

    for run_number, participant_number, target_scan_data_folder, target_scan_output_folder, target_scan_filename, \
            target_scan_gt_filename, gender, \
            palm_orientation, scan_type \
            in get_run_information(Path(root_folder, data_subfolder, 'Scan_Info.xlsx'), td_scan_data_folder,
                                   lidar_data_folder, zozo_data_folder, td_scan_output_folder, lidar_output_folder,
                                   zozo_output_folder):

        # Print details to screen
        print('\n\nrun_number:', run_number, 'participant_number:', participant_number, 'target_scan_data_folder:',
              target_scan_data_folder, 'target_scan_output_folder:', target_scan_output_folder,
              'target_scan_filename:', target_scan_filename, 'gender:', gender, 'palm_orientation:', palm_orientation,
              'scan_type:', scan_type, 'target_scan_gt_filename:', target_scan_gt_filename)
        print(f'runtime: {time() - start_time:.2f} time: {ctime()}')

        # If file doesn't exist, move to next run
        if target_scan_filename == '':
            continue

        # From the Scan_Info.xlsx file:
        # run_number: the current index into scan_info.xlsx file
        # participant_number: the participant number - can differ from run_number as the participant_number can
        # sometimes jump one if a participants results aren't used
        # target_scan_root_folder: the root folder of the scan. In this case we have a global root folder
        # ~\AUT 3D Scanner\Data collection AUT Millennium
        # and in this we have three folders, a folder for the 3D scanner, the Lidar and the zozo scans
        # target_scan_filename: the filename of the scan
        # gender: the gender of the participant (either 'male' or 'female' currently)
        # palm_orientation: either 'front' or 'side'
        # scan_type: either ScanType.td (for 3D scanner), ScanType.lidar or ScanType.zozo
        # target_scan_gt_filename: the name of the 'ground truth' scan which is also the 3D scanner scan with type
        # ScanType.td

        target_scan_filename_full = Path(target_scan_data_folder, target_scan_filename)  # get the target scan filename
        target_scan_name = target_scan_filename[:-4]  # get the name without the extension

        if not target_scan_output_folder.is_dir():
            target_scan_output_folder.mkdir(parents=True)

        # Load mesh or point_cloud target scan and rotate to STAR coordinate frame
        target_scan_trimesh = load_trimesh(target_scan_filename_full, scan_type)

        # Calculate the shape pc1 and the initial translation to shift the star model to match the target scan
        beta, star_initial_transl = get_shape_pc1_and_initial_translation(gender, target_scan_trimesh)

        # Get initial pose and constraints
        combined_parameters_numpy, combined_parameters_lower_limit_numpy, combined_parameters_upper_limit_numpy \
            = initial_pose(star_initial_transl, palm_orientation, run_number, scan_type, beta)

        # Get the STAR body model optimiser
        opto = StarBodyModelOptimiser(target_scan_trimesh,
                                      target_scan_name,
                                      scan_type,
                                      gender,
                                      combined_parameters_numpy,
                                      combined_parameters_lower_limit_numpy,
                                      combined_parameters_upper_limit_numpy,
                                      target_scan_output_folder,
                                      max_iters)

        # Run the fitting
        star_mesh_out_trimesh, star_mesh_out_zero_pose_trimesh = opto.run()

        # Filenames for saving
        star_mesh_out_filename = Path(target_scan_output_folder, f'{target_scan_name}_{max_iters:04}.obj')
        # with no hands or head
        star_mesh_out_no_hh_filename = Path(target_scan_output_folder, f'{target_scan_name}_{max_iters:04}_no_hh.obj')
        target_scan_no_hh_filename = Path(target_scan_output_folder, f'{target_scan_name}_no_hh.obj')
        # with no hands, head or feet
        star_mesh_out_no_hhf_filename = Path(target_scan_output_folder, f'{target_scan_name}_{max_iters:04}_no_hhf.obj')
        target_scan_no_hhf_filename = Path(target_scan_output_folder, f'{target_scan_name}_no_hhf.obj')

        # Save files with no hands, head or feet
        save_star_mesh_out_and_target_scan_with_no_hand_head_feet(scan_type, star_mesh_out_filename,
                                                                  star_mesh_out_no_hh_filename,
                                                                  star_mesh_out_no_hhf_filename,
                                                                  star_mesh_out_trimesh,
                                                                  target_scan_no_hh_filename,
                                                                  target_scan_no_hhf_filename,
                                                                  target_scan_trimesh)

        # Surface 2 surface results using gias
        results_gias[s2s_fitting_count_old, 0] = 'star fitting'
        results_gias[s2s_fitting_count_old, 1] = str(Path(*Path(target_scan_filename_full).parts[-3:]))
        results_gias[s2s_fitting_count_old, 2] = str(Path(*Path(star_mesh_out_filename).parts[-4:]))
        results_gias[s2s_fitting_count_old, 3:9] = np.array([x for x in surf2surf_dist(
            target_scan_filename_full, star_mesh_out_filename).values()])

        # Surface 2 surface results using 'compute_distance_metrics' function
        results_s2s_fitting_metrics_array[s2s_fitting_count, 0] = scan_type.name
        results_s2s_fitting_metrics_array[s2s_fitting_count, 1] = target_scan_filename
        results_s2s_fitting_metrics_array[s2s_fitting_count, 2] = '\\'.join(
            str(target_scan_filename_full).split('\\')[-3:-1])
        results_s2s_fitting_metrics_array[s2s_fitting_count, 3] = f'{target_scan_name}_{max_iters:04}.obj'
        results_s2s_fitting_metrics_array[s2s_fitting_count, 4] = '\\'.join(
            str(star_mesh_out_filename).split('\\')[-4:-1])
        results_s2s_fitting_metrics_array[s2s_fitting_count, 5:15] = \
            np.array([x for x in compute_distance_metrics(
                target_scan_filename_full,
                star_mesh_out_filename,
                sample_size_compared=n_points_in_pointcloud).values()])
        results_s2s_fitting_metrics_array[s2s_fitting_count, 15:25] = \
            np.array([x for x in compute_distance_metrics(
                target_scan_no_hh_filename,
                star_mesh_out_no_hh_filename,
                sample_size_compared=n_points_in_pointcloud).values()])
        results_s2s_fitting_metrics_array[s2s_fitting_count, 25:35] = \
            np.array([x for x in compute_distance_metrics(
                target_scan_no_hhf_filename,
                star_mesh_out_no_hhf_filename,
                sample_size_compared=n_points_in_pointcloud).values()])

        s2s_fitting_count += 1
        s2s_fitting_count_old += 1

        # Save zero pose star mesh
        star_mesh_out_zero_pose_filename = \
            Path(target_scan_output_folder, f'{target_scan_name}_{max_iters:04}_zero_pose.obj')
        save_trimesh(star_mesh_out_zero_pose_trimesh, ScanType.star, star_mesh_out_zero_pose_filename)

        # Save a CloudCompare sampled point cloud of the star meshes above
        save_cloud_compare_sampled_point_cloud(star_mesh_out_filename)
        save_cloud_compare_sampled_point_cloud(star_mesh_out_zero_pose_filename)

        # Compare star fitted mesh with original mesh/pointcloud
        if scan_type == ScanType.lidar:
            # Need to load lidar mesh
            target_mesh_filename = append_string_to_filename(target_scan_filename_full, '_mesh', '.obj')
            target_point_cloud_filename = target_scan_filename_full
        else:
            target_mesh_filename = target_scan_filename_full
            target_point_cloud_filename = append_string_to_filename(target_scan_filename_full,
                                                                    '_sampled_pointcloud', '.ply')

        star_mesh_out_sampled_point_cloud_filename = append_string_to_filename(star_mesh_out_filename,
                                                                               '_SAMPLED_POINTS', '.bin')

        # Calculate cloud compare metrics
        if target_mesh_filename.exists():
            # Calculate cloud-to-mesh results
            mean, stdev, m_s_string = calc_cloud_compare_c2m_dist(star_mesh_out_filename,
                                                                  target_mesh_filename)
            results_c2m[scan_type.value, run_number] = m_s_string

        # Calculate cloud-to-cloud results
        mean, stdev, m_s_string = calc_cloud_compare_c2c_dist(star_mesh_out_sampled_point_cloud_filename,
                                                              target_point_cloud_filename)
        results_c2c[scan_type.value, run_number] = m_s_string

        # Compare the zero pose between lidar/zozo and GT
        if (scan_type == ScanType.lidar or scan_type == ScanType.zozo) and target_scan_gt_filename != '':
            # calculate the target_scan_zero_pose_gt_filename filename
            target_scan_gt_name = target_scan_gt_filename[:-4]
            target_scan_zero_pose_gt_filename = Path(td_scan_output_folder,
                                                     f'{target_scan_gt_name}_{max_iters:04}_zero_pose.obj')

            # Calculate the zero-pose cloud-to-mesh results
            mean, stdev, m_s_string = calc_cloud_compare_c2m_dist(star_mesh_out_zero_pose_filename,
                                                                  target_scan_zero_pose_gt_filename)
            results_c2m[scan_type.value + 2, run_number] = m_s_string

            target_point_cloud_filename = append_string_to_filename(target_scan_zero_pose_gt_filename,
                                                                    "_SAMPLED_POINTS", ".bin")
            filename_source_point_cloud = append_string_to_filename(star_mesh_out_zero_pose_filename,
                                                                    "_SAMPLED_POINTS", ".bin")

            # Calculate zero-pose cloud-to-cloud results
            mean, stdev, m_s_string = calc_cloud_compare_c2c_dist(filename_source_point_cloud,
                                                                  target_point_cloud_filename)
            results_c2c[scan_type.value + 2, run_number] = m_s_string

            # Zero-pose surface 2 surface results using gias
            results_gias[s2s_fitting_count_old, 0] = 'zero-pose fitting'
            results_gias[s2s_fitting_count_old, 1] = str(
                Path(*Path(target_scan_zero_pose_gt_filename).parts[-3:]))
            results_gias[s2s_fitting_count_old, 2] = str(
                Path(*Path(star_mesh_out_zero_pose_filename).parts[-3:]))
            results_gias[s2s_fitting_count_old, 3:9] = np.array(
                [x for x in surf2surf_dist(target_scan_zero_pose_gt_filename,
                                           star_mesh_out_zero_pose_filename).values()])

            # Zero-pose surface 2 surface results using 'compute_distance_metrics' function
            results_s2s_zero_pose_metrics_array[s2s_zero_pose_count, 0] = scan_type.name
            results_s2s_zero_pose_metrics_array[
                s2s_zero_pose_count, 1] = f'{target_scan_gt_name}_{max_iters:04}_zero_pose.obj'
            results_s2s_zero_pose_metrics_array[s2s_zero_pose_count, 2] = '\\'.join(
                str(target_scan_zero_pose_gt_filename).split('\\')[-4:-1])
            results_s2s_zero_pose_metrics_array[
                s2s_zero_pose_count, 3] = f'{target_scan_name}_{max_iters:04}_zero_pose.obj'
            results_s2s_zero_pose_metrics_array[s2s_zero_pose_count, 4] = '\\'.join(
                str(star_mesh_out_zero_pose_filename).split('\\')[-4:-1])
            results_s2s_zero_pose_metrics_array[s2s_zero_pose_count, 5:15] = \
                np.array([x for x in compute_distance_metrics(
                    target_scan_zero_pose_gt_filename,
                    star_mesh_out_zero_pose_filename,
                    sample_size_reference=n_points_in_pointcloud,
                    sample_size_compared=n_points_in_pointcloud).values()])

            s2s_zero_pose_count += 1
            s2s_fitting_count_old += 1

    # Save the results arrays to Excel spreadsheets
    # Create dataframes of the results
    df_gias = pd.DataFrame(results_gias, index=None)
    df_c2c = pd.DataFrame(results_c2c, index=None)
    df_c2m = pd.DataFrame(results_c2m, index=None)

    results_s2s_xlsx_filename = Path(output_root_folder,
                                     f'{star_fitting_results_subfolder}.xlsx')
    results_s2s_xlsx_filename_old = Path(output_root_folder,
                                         f'{star_fitting_results_subfolder}_old.xlsx')

    # Save the dataframes to Excel files
    with pd.ExcelWriter(results_s2s_xlsx_filename_old) as writer:
        df_gias.to_excel(writer, sheet_name='gias', index=False, header=False)
        df_c2c.to_excel(writer, sheet_name='c2c', index=False, header=False)
        df_c2m.to_excel(writer, sheet_name='c2m', index=False, header=False)

    df1 = pd.DataFrame(results_s2s_fitting_metrics_array, index=None)
    df2 = pd.DataFrame(results_s2s_zero_pose_metrics_array, index=None)
    with pd.ExcelWriter(results_s2s_xlsx_filename) as writer:
        df1.to_excel(writer, sheet_name='Scan fitting', index=False, header=False)
        df2.to_excel(writer, sheet_name='Zero pose comparison', index=False, header=False)

    print(f'total runtime: {time() - start_time:.2f} time: {ctime()}')
    print('Finished!')


def create_results_arrays():
    """
    Create the results numpy arrays for the cloud-to-mesh and cloud-to-cloud results and populate the headers
    """
    results_c2m = np.empty(shape=(6, 38), dtype='object')
    results_c2c = np.empty(shape=(6, 38), dtype='object')
    run_labels_list = [f'R{x + 1:03}' for x in range(37)]
    results_c2m[0, 1:] = run_labels_list
    results_c2m[1:, 0] = ['Ground Truth', 'Lidar', 'Zozo', 'Lidar-vs-GT', 'Zozo-vs-GT']
    results_c2c[0, 1:] = run_labels_list
    results_c2c[1:, 0] = ['Ground Truth', 'Lidar', 'Zozo', 'Lidar-vs-GT', 'Zozo-vs-GT']
    results_gias = np.empty(shape=(300, 100), dtype='object')
    results_gias[0, 0:9] = ['Scan Type', 'Reference Filename', 'Compared Filename', 'jaccard', 'dice', 'dmax', 'drms',
                            'dmean', 'dhausdorff']
    results_s2s_fitting_metrics_array = np.empty(shape=(100, 100), dtype='object')
    results_s2s_zero_pose_metrics_array = np.empty(shape=(100, 100), dtype='object')
    results_s2s_fitting_metrics_array[0, 0:35] = ['Scan Type', 'Filename', 'Folder', 'STAR Filename', 'STAR Folder',
                                                  'rmse', 'rmse_sigma', 'rmse_min', 'rmse_max',
                                                  'mean', 'mean_sigma', 'min', 'max',
                                                  'mean_cc', 'mean_sigma_cc',
                                                  'rmse_no_hh', 'rmse_sigma_no_hh', 'rmse_min_no_hh', 'rmse_max_no_hh',
                                                  'mean_no_hh', 'mean_sigma_no_hh', 'min_no_hh', 'max_no_hh',
                                                  'mean_cc_no_hh', 'mean_sigma_cc_no_hh',
                                                  'rmse_no_hhf', 'rmse_sigma_no_hhf', 'rmse_min_no_hhf',
                                                  'rmse_max_no_hhf',
                                                  'mean_no_hhf', 'mean_sigma_no_hhf', 'min_no_hhf', 'max_no_hhf',
                                                  'mean_cc_no_hhf', 'mean_sigma_cc_no_hhf']
    results_s2s_zero_pose_metrics_array[0, 0:15] = ['Scan Type', 'Reference Filename (GT zero-pose)', 'Reference Path',
                                                    'Compared Filename (Scan zero-pose)', 'Compared Path', 'rmse',
                                                    'rmse_sigma',
                                                    'rmse_min', 'rmse_max', 'mean', 'mean_sigma', 'min', 'max',
                                                    'mean_cc', 'mean_sigma_cc']
    return results_c2c, results_c2m, results_gias, results_s2s_fitting_metrics_array, \
        results_s2s_zero_pose_metrics_array


def read_config_file(config_filename: Path) -> Tuple[str, str, str, str, str, str, str, int, int, str, str]:
    """
    Reads a configuration file and return the values

    :param config_filename: the name of the config file
    """

    if config_filename.exists():
        print(f'Config file {config_filename} exists')
    else:
        print(f'Config file {config_filename} DOESN\'T exist')

    # config_filename = config_filename[config_filename.rfind("/")+1:]

    config = configobj.ConfigObj(infile=str(config_filename), raise_errors=True, unrepr=True)

    root_folder = config['root_folder']

    data_subfolder = config['data_subfolder']

    output_subfolder = config['output_subfolder']

    td_scan_subfolder = config['td_scan_subfolder']

    lidar_subfolder = config['lidar_subfolder']

    zozo_subfolder = config['zozo_subfolder']

    star_fitting_results_folder_name = config['star_fitting_results_subfolder']

    max_iters = config['max_iters']

    n_points_in_pointcloud = config['n_points_in_pointcloud']

    cloud_compare_path = config['cloud_compare_path']

    cloud_compare_tmp_folder = config['cloud_compare_tmp_folder']

    return root_folder, data_subfolder, output_subfolder, td_scan_subfolder, lidar_subfolder, zozo_subfolder, \
        star_fitting_results_folder_name, max_iters, n_points_in_pointcloud, cloud_compare_path, \
        cloud_compare_tmp_folder


def get_run_information(scan_info_filename: Path, td_scan_data_folder: Path, lidar_data_folder: Path,
                        zozo_data_folder: Path, td_scan_output_folder: Path, lidar_output_folder: Path,
                        zozo_output_folder: Path):
    """

    Get information regarding the scans we are fitting with the STAR model from the 'Scan_Info.xlsx' file.

    :param scan_info_filename: the path to the Scan_Info.xlsx file
    :param td_scan_data_folder: the 3d scans data folder
    :param lidar_data_folder: the lidar scans data folder
    :param zozo_data_folder: the zozo scans data folder
    :param td_scan_output_folder: the 3d scans results folder
    :param lidar_output_folder: the lidar scans results folder
    :param zozo_output_folder: the zozo scans results folder
    """
    scan_info = pd.read_excel(scan_info_filename, keep_default_na=False)

    number_runs = scan_info.shape[0]

    for run_number in scan_info.Run_Number:
        if run_number > number_runs:
            print(f'run_number > Rows of data in scan_info file: {scan_info_filename}')
            print('Now exiting...')
            exit(0)
        participant_number = scan_info.Participant_Number[run_number - 1]
        gender = scan_info.Gender[run_number - 1]
        target_scan_filename = None
        target_scan_data_folder = None
        target_scan_output_folder = None
        scan_palm_orientation = None
        target_scan_gt_filename = scan_info.GT_Filename[run_number - 1]

        for scan_type in [ScanType.td, ScanType.lidar, ScanType.zozo]:
            if scan_type == ScanType.td:
                target_scan_filename = scan_info.GT_Filename[run_number - 1]
                target_scan_data_folder = td_scan_data_folder
                target_scan_output_folder = td_scan_output_folder
                scan_palm_orientation = scan_info.GT_Palm_Orientation[run_number - 1]
            elif scan_type == ScanType.lidar:
                target_scan_filename = scan_info.Lidar_Filename[run_number - 1]
                target_scan_data_folder = lidar_data_folder
                target_scan_output_folder = lidar_output_folder
                scan_palm_orientation = scan_info.Lidar_Palm_Orientation[run_number - 1]
            elif scan_type == ScanType.zozo:
                target_scan_filename = scan_info.Zozo_Filename[run_number - 1]
                target_scan_data_folder = zozo_data_folder
                target_scan_output_folder = zozo_output_folder
                scan_palm_orientation = scan_info.Zozo_Palm_Orientation[run_number - 1]

            yield run_number, participant_number, \
                target_scan_data_folder, target_scan_output_folder, \
                target_scan_filename, target_scan_gt_filename, \
                gender, scan_palm_orientation, scan_type,


if __name__ == "__main__":
    main()
