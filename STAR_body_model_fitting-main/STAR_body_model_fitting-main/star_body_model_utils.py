"""
star_body_model_utils.py
by Conrad Werkhoven, Auckland Bioengineering Institute

https://github.com/ConradW01/STAR_body_model_fitting

An assortment of utility scripts used in the STAR body model fitting

"""
import json
import math
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, Union, List

import numpy as np
import open3d as o3d
import pytorch3d
import torch
import trimesh
from PIL import Image, ImageFont, ImageDraw
from matplotlib.colors import LinearSegmentedColormap
from pyquaternion import Quaternion
from scipy.spatial import KDTree
from trimesh import Trimesh, PointCloud
from trimesh.caching import TrackedArray

from star_body_model_surface_to_surface_distance import dim_unit_scaling, calcSegmentationErrors

# A list of the joints in the STAR model
STAR_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand']

# ANSI color codes for printing colored text to screen
RED = '\033[31m'
BLACK = '\033[30m'

# Trimesh and Open3D colors
trimesh_color_light_green = [128, 255, 128, 255]
trimesh_color_light_grey = [192, 192, 192, 255]
trimesh_color_star_grey = [150, 150, 150, 255]
trimesh_color_white = [255, 255, 255, 255]
o3d_color_light_red = [1.0, 0.5, 0.5]
o3d_color_light_green = [0.5, 1.0, 0.5]
o3d_color_light_blue = [0.0, 0.5, 1.0]
o3d_color_red = [1, 0, 0]
o3d_color_green = [0, 1, 0]
o3d_color_blue = [0, 0, 1]
o3d_color_light_grey = [0.75, 0.75, 0.75]
o3d_color_star_grey = [0.59, 0.59, 0.59]
o3d_color_gold = [1, 0.706, 0]
o3d_color_white = [1, 1, 1]

# deg2rad conversion constants
deg2rad = 1.0 / 180.0 * math.pi
rad2deg = 180.0 / math.pi

# Transformation matrices between m and mm
mm_to_m_transform_matrix = np.array([[0.001, 0, 0, 0], [0, 0.001, 0, 0], [0, 0.0, 0.001, 0], [0, 0, 0, 1]])
m_to_mm_transform_matrix = np.array([[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1]])

# Transformation matrices between different coordinate frames
ct_scan_to_star_transform_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
td_scan_to_star_transform_matrix = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
lidar_to_star_transform_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
zozo_to_star_transform_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


# Type of scan - this is used to calculate the rotation transformation when loading and saving the mesh
class ScanType(Enum):
    td = 1  # aut 3d scan
    lidar = 2  # lidar scan
    zozo = 3  # zozo scan
    star = 4  #
    ct = 5  # ct scan in mm
    ct_meters = 6  # ct scan in meters


def trimesh_to_torch(scan: Union[Trimesh, PointCloud]) -> \
        Union[pytorch3d.structures.Meshes, pytorch3d.structures.Pointclouds]:
    if isinstance(scan, trimesh.base.Trimesh):
        verts = torch.FloatTensor(np.asarray(scan.vertices))
        faces = torch.IntTensor(np.asarray(scan.faces))
        return pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
    elif isinstance(scan, trimesh.points.PointCloud):
        verts = torch.FloatTensor(np.asarray(scan.vertices))
        return pytorch3d.structures.Pointclouds(points=[verts])


def trimesh_to_open3d(scan: Union[Trimesh, PointCloud]) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    if isinstance(scan, trimesh.base.Trimesh):
        return scan.as_open3d
    elif isinstance(scan, trimesh.points.PointCloud):
        points = o3d.utility.Vector3dVector(scan.vertices)
        pointcloud_o3d = o3d.geometry.PointCloud(points=points)
        try:
            colors = o3d.utility.Vector3dVector(np.array(scan.colors)[:, :3] / 255)
            pointcloud_o3d.colors = colors
        except IndexError:
            print('No colors in pointcloud')
            pointcloud_o3d.paint_uniform_color(o3d_color_light_red)
        return pointcloud_o3d


def print_out_torch_info() -> None:
    """
    Print out pytorch info to screen
    """
    print('torch version:', torch.__version__)
    print('pytorch3d version:', pytorch3d.__version__)
    print('torch.cuda.is_available:', torch.cuda.is_available())
    print(os.environ.get('CUDA_PATH'))


def get_shape_pc1_and_initial_translation(gender: str, target_scan_trimesh: Union[Trimesh, PointCloud]) -> \
        Tuple[float, np.ndarray]:
    """
    Calculate the first principal component value that corresponds to the height of the scan and also calculate
    the initial translation of the center of the scan relative to the STAR body model
    :param gender: either 'male' or 'female'
    :param target_scan_trimesh: the target scan
    :return: the first principal component and the initial translation
    """
    vertices = target_scan_trimesh.vertices
    star_initial_transl = np.array(
        [np.mean(vertices[:, 0]), np.min(vertices[:, 1]) + 1.3, np.mean(vertices[:, 2]) - 0.02])
    # Get the height
    height = float(np.max(vertices[:, 1])) - float(np.min(vertices[:, 1]))
    beta = None
    # For STAR model - work out the value of the first shape component
    if gender == 'male':
        beta = 13.021 * height - 23.165
    elif gender == 'female':
        beta = 15.291 * height - 25.141
    print(f'height: {height} beta: {beta}')
    print(f'star_initial_transl: {star_initial_transl}')
    return beta, star_initial_transl


def surf2surf_dist(poly_truth_filename_full: Path,
                   poly_pred_filename_full: Path,
                   scan_type: ScanType = None) -> Dict:
    """
    Measures the surf-to-surf distance between two meshes

    :param poly_truth_filename_full: full path and filename to the truth poly
    :param poly_pred_filename_full: full path and filename to the pred poly
    :param scan_type:
    :return: a dict of various distance measures between the two polys.

    Calculates and returns a dict of a number of distance measures between the truth and the predicted meshes
    """

    print('Starting surf2surf_dist')

    img_spacing = np.array([0.01, ] * 3, dtype=float)
    default_unit = 'm'
    if scan_type == ScanType.ct:
        ground_truth_unit = 'mm'
    else:
        ground_truth_unit = 'm'
    test_unit = 'm'

    gt_unit_scaling = dim_unit_scaling(ground_truth_unit, default_unit)
    test_unit_scaling = dim_unit_scaling(test_unit, default_unit)

    # check files exist
    if Path(poly_truth_filename_full).exists() and Path(poly_pred_filename_full).exists():
        print('In surf2surf_dist - polys exist')

        results, surf_test, surf_gt, img_test, img_gt = calcSegmentationErrors(
            str(poly_pred_filename_full),
            str(poly_truth_filename_full),
            img_spacing,
            gt_unit_scaling,
            test_unit_scaling)

        return results
    else:
        print('In surf2surf_dist - polys DONT exist')
        return {}


def compute_distance_metrics(mesh_reference: Union[Path, Trimesh, PointCloud, np.ndarray],
                             mesh_compared: Union[Path, Trimesh, PointCloud, np.ndarray],
                             sample_size_reference: int = 200000, sample_size_compared: int = 200000) -> Dict:
    """
    Computes various surface-to-surface distance metrics between two meshes and returns the values in a dictionary

    :param mesh_reference:
    :param mesh_compared:
    :param sample_size_reference:
    :param sample_size_compared:
    :return: rmse, rmse_sigma, rmse_min, rmse_max, mean, mean_sigma, min, max, mean_cloudcompare,
    mean_sigma_cloudcompare
    """
    warning_str = 'Please pass either a path to a mesh or a Trimesh or PointCloud or numpy array'

    vertices_1 = None
    vertices_2 = None
    if isinstance(mesh_reference, Path):
        mesh_reference = trimesh.load(mesh_reference, process=False, maintain_order=True)
    if isinstance(mesh_reference, Trimesh) or isinstance(mesh_reference, Trimesh):
        if sample_size_reference is None:
            vertices_1 = np.array(mesh_reference.vertices)
        else:
            # vertices_1, sampled_index = trimesh.sample.sample_surface(mesh_reference, sample_size_reference)
            vertices_1, sampled_index = trimesh.primitives.sample.sample_surface(mesh_reference, sample_size_reference)
    elif isinstance(mesh_reference, PointCloud) or isinstance(mesh_reference, PointCloud):
        if sample_size_reference is None:
            print('Warning - sampling point cloud not implemented for compute_distance_matrix')
        vertices_1 = np.array(mesh_reference.vertices)
    elif isinstance(mesh_reference, np.ndarray):
        vertices_1 = mesh_reference
    else:
        print(warning_str)
    if isinstance(mesh_compared, Path):
        mesh_compared = trimesh.load(mesh_compared, process=False, maintain_order=True)
    if isinstance(mesh_compared, Trimesh) or isinstance(mesh_compared, Trimesh):
        if sample_size_compared is None:
            vertices_2 = np.array(mesh_compared.vertices)
        else:
            # vertices_2, sampled_index = trimesh.sample.sample_surface(mesh_compared, sample_size_compared)
            vertices_2, sampled_index = trimesh.primitives.sample.sample_surface(mesh_compared, sample_size_compared)
    elif isinstance(mesh_compared, PointCloud) or isinstance(mesh_compared, PointCloud):
        if sample_size_compared is None:
            print('Warning - sampling point cloud not implemented for compute_distance_matrix')
        vertices_2 = np.array(mesh_compared.vertices)  # ??????? check this
    elif isinstance(mesh_compared, np.ndarray):
        if sample_size_compared is not None:
            print('Warning - sampling numpy array not implemented for compute_distance_matrix')
        vertices_2 = mesh_compared
    else:
        print(warning_str)

    # If in mm convert to m
    if (np.max(vertices_1[2]) - np.min(vertices_1[2])) > 1000:
        vertices_1 = vertices_1 / 1000
    if (np.max(vertices_2[2]) - np.min(vertices_2[2])) > 1000:
        vertices_2 = vertices_2 / 1000

    # k nearest neighbour search
    tree1 = KDTree(vertices_1)
    tree2 = KDTree(vertices_2)
    d21, d21_index = tree1.query(vertices_2, k=1)
    d12, d12_index = tree2.query(vertices_1, k=1)
    d = np.hstack([d21, d12])

    # c.f. gias
    rmse = np.sqrt(np.mean(d * d))
    rmse_sigma = np.sqrt(np.std(d * d))
    rmse_min = np.sqrt(np.min(d * d))
    rmse_max = np.sqrt(np.max(d * d))

    mean_ = np.mean(np.sqrt(d * d))
    mean_sigma = np.std(np.sqrt(d * d))
    min_ = np.min(np.sqrt(d * d))
    max_ = np.max(np.sqrt(d * d))

    # c.f. cloud compare
    d = d21
    mean_cc = np.mean(np.sqrt(d * d))
    mean_sigma_cc = np.std(np.sqrt(d * d))

    metric_dict = {'rmse': rmse, 'rmse_sigma': rmse_sigma, 'rmse_min': rmse_min, 'rmse_max': rmse_max, 'mean': mean_,
                   'mean_sigma': mean_sigma, 'min': min_, 'max': max_, 'mean_cc': mean_cc,
                   'mean_sigma_cc': mean_sigma_cc}

    return metric_dict


def rotate_to_star_frame(scan: Union[Trimesh, PointCloud], scan_type: ScanType) -> Union[Trimesh, PointCloud]:
    """
    Rotates a scan to the STAR coordinate frame

    :param scan: A Trimesh or PointCloud object
    :param scan_type: type of scan from ScanType
    """
    if scan_type == ScanType.ct or scan_type == ScanType.ct_meters:
        scan.apply_transform(ct_scan_to_star_transform_matrix)
    elif scan_type == ScanType.td:
        scan.apply_transform(td_scan_to_star_transform_matrix)
    elif scan_type == ScanType.lidar:
        scan.apply_transform(lidar_to_star_transform_matrix)
    elif scan_type == ScanType.zozo:
        scan.apply_transform(zozo_to_star_transform_matrix)
    return scan


def rotate_to_original_frame(scan: Union[Trimesh, PointCloud], scan_type: ScanType) -> Union[Trimesh, PointCloud]:
    """
    Rotates a scan to from STAR coordinate frame to its original coordinate frame

    :param scan: A Trimesh or PointCloud object
    :param scan_type: type of scan from ScanType
    """
    if scan_type == ScanType.ct or scan_type == ScanType.ct_meters:
        scan.apply_transform(np.linalg.inv(ct_scan_to_star_transform_matrix))
    elif scan_type == ScanType.td:
        scan.apply_transform(np.linalg.inv(td_scan_to_star_transform_matrix))
    elif scan_type == ScanType.lidar:
        scan.apply_transform(np.linalg.inv(lidar_to_star_transform_matrix))
    elif scan_type == ScanType.zozo:
        scan.apply_transform(np.linalg.inv(zozo_to_star_transform_matrix))
    return scan


def resize_from_gias_mm_to_m(scan, scan_type: ScanType) -> Union[Trimesh, PointCloud]:
    """
    Resize a Trimesh or PointCloud from mm to m

    :param scan: A Trimesh or PointCloud object
    :param scan_type: type of scan from ScanType
    """
    if scan_type == ScanType.ct:
        scan.apply_transform(mm_to_m_transform_matrix)
    return scan


def resize_to_gias_m_to_mm(scan, scan_type: ScanType) -> Union[Trimesh, PointCloud]:
    """
    Resize a Trimesh or PointCloud from m to mm

    :param scan: A Trimesh or PointCloud object
    :param scan_type: type of scan from ScanType
    """
    if scan_type == ScanType.ct:
        scan.apply_transform(m_to_mm_transform_matrix)
    return scan


def load_trimesh(target_filename: Path, scan_type: ScanType, rotate_to_star_frame_flag: bool = True) -> \
        Union[Trimesh, PointCloud]:
    """
    Load a scan and rotate it to the STAR coordinate frame

    :param target_filename: A path to the file
    :param scan_type: type of scan from ScanType
    :param rotate_to_star_frame_flag: bool True if we want to rotate scan to STAR frame
    :return: either a Trimesh or a PointCloud
    """
    trimesh_object = trimesh.load(target_filename, process=False, maintain_order=True)
    if scan_type == ScanType.ct:
        trimesh_object = resize_from_gias_mm_to_m(trimesh_object, scan_type)
    if rotate_to_star_frame_flag:
        trimesh_object = rotate_to_star_frame(trimesh_object, scan_type)

    return trimesh_object


def save_trimesh(mesh_out_trimesh: Union[Trimesh, PointCloud], scan_type: ScanType, output_filename: Path) -> None:
    """
    Save a Trimesh to drive. The saved mesh is rotated to the original coordinate system of the scan scan_type

    :param mesh_out_trimesh: A scan to save to disk
    :param scan_type: type of scan from ScanType
    :param output_filename: The filename of the output mesh
    """
    if scan_type != ScanType.star:
        # If not star mesh rotate to original frame
        mesh_out_trimesh = rotate_to_original_frame(mesh_out_trimesh, scan_type)
    if scan_type == ScanType.ct:
        # If scantype is ct (which is originally in mm but when loaded we converted it to m) convert back from m
        # to the original mm
        mesh_out_trimesh = resize_to_gias_m_to_mm(mesh_out_trimesh, scan_type)
    if isinstance(mesh_out_trimesh, Trimesh):
        # If it's a mesh calculate vertex normals and save
        vn = mesh_out_trimesh.vertex_normals
        # We need to calculate these so that they are cached else they will not be saved
        mesh_out_trimesh.export(str(output_filename), include_normals=True)
    elif isinstance(mesh_out_trimesh, PointCloud):
        # If it's a pointcloud then just save
        mesh_out_trimesh.export(str(output_filename))


def create_open3d_mesh_from_verts_and_faces(verts: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Create an open3d mesh from numpy arrays of vertices and faces

    :param verts: a numpy array of the vertices
    :param faces: a numpy array of the faces
    :return: An open3d mesh
    """
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = verts
    mesh_o3d.triangles = faces
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


def is_number(x: Union[float, str]) -> bool:
    """
    Test whether object is a number

    :param x: an object to be tested
    :return: True if a number, False otherwise
    """
    try:
        num = float(x)
        # check for "nan" floats
        isnumber = num == num  # or use `math.isnan(num)`
    except ValueError:
        isnumber = False
    return isnumber


def get_mean_stdev_from_cc_log(filename: Path) -> Tuple[float, float, str]:
    """
    Get the mean and stddev from a cloud compare log file. The log file is typically created after using
    calc_cloud_compare_c2m_dist or calc_cloud_compare_c2c_dist to calculate the c2c or c2m dist

    :param filename: a cloud compare log file
    :return: the mean and standard deviation as floats, and also together as a string
    """
    for line in open(filename, 'r'):
        if "Mean distance" in line:
            m, s = [float(i) for i in line.split() if is_number(i)]
            m_s_string = f'{m:5.4f}\u00B1{s:5.4f}'
            return m, s, m_s_string


def append_string_to_filename(filename: Path,
                              append_str: str = "",
                              suffix_str: str = "") -> Path:
    """
    Append a string to a filename. so that 'name.suffix' becomes name_(append_str).(suffix_str)

    :param filename: the original filename as a pathlib Path
    :param append_str: text that will be appended to the end of the string, but before the extension.
    :param suffix_str: the extension name if given else keep old extension name

    :return: the new filename
    """
    if suffix_str is None:
        suffix_str = filename.suffix
        return Path(f'{filename.with_suffix("")}{append_str}').with_suffix(suffix_str)
    else:
        return Path(f'{filename.with_suffix("")}{append_str}').with_suffix(suffix_str)


def remove_string_from_filename(filename: Path,
                                remove_str: str = "",
                                suffix_str: str = "") -> Path:
    """
    Remove a string from the end of a filename

    :param filename: the original filename as a pathlib Path
    :param remove_str: text to remove from the filename
    :param suffix_str: the extension name if given else keep old extension name

    :return: the new filename
    """
    filename_without_suffix = filename.with_suffix("")
    if str(filename_without_suffix)[-len(remove_str):] == remove_str:
        filename_without_suffix = Path(str(filename_without_suffix)[:-len(remove_str)])
    else:
        print(f'Warning: {filename} does not contain string {remove_str} so cannot be removed from end of string in'
              f' function remove_string_from_filename')
    if suffix_str is None:
        suffix_str = filename.suffix
        return Path(f'{filename_without_suffix}').with_suffix(suffix_str)
    else:
        return Path(f'{filename_without_suffix}').with_suffix(suffix_str)


def calc_cloud_compare_c2m_dist(
        source_filename: Path,
        target_filename: Path,
        cloudcomparepath: Path = Path("C:/Program Files/CloudCompare/CloudCompare"),
        cloudcomparetmpfolder: Path = Path("D:/tmp/star")) -> Tuple[float, float, str]:
    """
    Compute the cloud compare cloud-to-mesh distance

    :param source_filename: source filename
    :param target_filename: target filename
    :param cloudcomparepath: path to cloud compare installation
    :param cloudcomparetmpfolder: path to a tmp folder for cloud compare batch files
    :return: the mean and standard deviation as floats, and also together as a string
    """

    # Test the files exist
    if source_filename.exists():
        print(f'source_filename {str(source_filename)} exists')
    else:
        print(f'source_filename {str(source_filename)} DOESNT exist')
    if target_filename.exists():
        print(f'target_filename {str(target_filename)} exists')
    else:
        print(f'target_filename {str(target_filename)} DOESNT exist')

    if not cloudcomparetmpfolder.is_dir():
        cloudcomparetmpfolder.mkdir(parents=True)

    cc_log_filename = append_string_to_filename(source_filename, '_CC_c2m', '.log')
    bat_string = f'"{cloudcomparepath}" -SILENT ' \
                 f'-o "{str(source_filename)}" ' \
                 f'-o "{str(target_filename)}" ' \
                 f'-LOG_FILE "{str(cc_log_filename)}" ' \
                 f'-NO_TIMESTAMP -c2m_dist'
    with open(f'{str(cloudcomparetmpfolder)}/CloudCompare_c2m.bat', 'w', encoding='utf-8') as f_object:
        f_object.write(bat_string)
        f_object.write('\n')
    subprocess.call([f'{str(cloudcomparetmpfolder)}/CloudCompare_c2m.bat'])

    c2m_dist_source_filename = append_string_to_filename(source_filename, '_C2M_DIST', '.bin')
    # mesh_source_filename = source_filename

    # Interpolate the c2m_dist point cloud back onto the mesh vertices and save the vertices with the
    sf_interp_log_filename = append_string_to_filename(source_filename, '_CC_c2m_sf_interp', '.log')
    bat_string = f'"{str(cloudcomparepath)}" -SILENT -NO_TIMESTAMP -AUTO_SAVE OFF ' \
                 f'-o "{str(c2m_dist_source_filename)}" ' \
                 f'-EXTRACT_VERTICES ' \
                 f'-LOG_FILE "{str(sf_interp_log_filename)}" ' \
                 f'-C_EXPORT_FMT PLY -PLY_EXPORT_FMT ASCII -SAVE_CLOUDS ' \
                 f''

    with open(f'{str(cloudcomparetmpfolder)}/CloudCompare_interp.bat', 'w', encoding='utf-8') as f_object:
        f_object.write(bat_string)
        f_object.write('\n')
    subprocess.call([f'{str(cloudcomparetmpfolder)}/CloudCompare_interp.bat'])

    # Load mesh vertices with scalar_C2C_absolute_distances
    mesh_vertices_source_filename = Path(str(c2m_dist_source_filename.with_suffix("")) + '.vertices.ply')

    mesh_vertices_source = trimesh.load(mesh_vertices_source_filename, process=False, maintain_order=True)
    scalar_c2m_signed_distances = mesh_vertices_source.metadata['_ply_raw']['vertex']['data'][
        'scalar_C2M_signed_distances'].squeeze()

    # Normalize the scale
    max_scalar = np.max(scalar_c2m_signed_distances)
    min_scalar = np.min(scalar_c2m_signed_distances)
    scalar_c2m_signed_distances = (scalar_c2m_signed_distances - min_scalar) / (max_scalar - min_scalar)
    # max_scalar2 = np.max(scalar_c2m_signed_distances)
    # min_scalar2 = np.min(scalar_c2m_signed_distances)

    # Load the mesh
    mesh_source = trimesh.load(source_filename, process=False, maintain_order=True)

    my_cmap = cc_bluegreenyellowred_colormap()

    # Set the vertex colors according to c2c_dist using the cc colormap
    mesh_source.visual.vertex_colors = my_cmap(scalar_c2m_signed_distances)

    mesh_source_filename_saved = append_string_to_filename(source_filename, '_C2M_DIST_SURFACE', '.ply')
    # Save the mesh to a file in ply format
    # trimesh.exchange.export.export_mesh(mesh_source, mesh_source_filename_saved)
    trimesh.base.export_mesh(mesh_source, mesh_source_filename_saved)

    return get_mean_stdev_from_cc_log(cc_log_filename)


def calc_cloud_compare_c2c_dist(source_filename: Path,
                                target_filename: Path,
                                cloudcomparepath: Path = Path("C:/Program Files/CloudCompare/CloudCompare"),
                                cloudcomparetmpfolder: Path = Path("D:/tmp/star")) -> Tuple[float, float, str]:
    """
    Compute the cloud compare cloud-to-cloud distance

    :param source_filename: source filename
    :param target_filename: target filename
    :param cloudcomparepath: path to cloud compare installation
    :param cloudcomparetmpfolder: path to a tmp folder for cloud compare batch files
    :return: the mean and standard deviation as floats, and also together as a string

    """
    # Test the files exist
    if source_filename.exists():
        print(f'source_filename {str(source_filename)} exists')
    else:
        print(f'source_filename {str(source_filename)} DOESNT exist')
    if target_filename.exists():
        print(f'target_filename {str(target_filename)} exists')
    else:
        print(f'target_filename {str(target_filename)} DOESNT exist')

    if not cloudcomparetmpfolder.is_dir():
        cloudcomparetmpfolder.mkdir(parents=True)

    # Get the c2c_dist between the original scan and the fitted
    cc_log_filename = append_string_to_filename(source_filename, '_CC_c2c', '.log')
    bat_string = f'"{str(cloudcomparepath)}" -SILENT ' \
                 f'-o "{str(source_filename)}" ' \
                 f'-o "{str(target_filename)}" ' \
                 f'-LOG_FILE "{str(cc_log_filename)}" ' \
                 f'-NO_TIMESTAMP -c2c_dist'
    f''
    with open(f'{str(cloudcomparetmpfolder)}/CloudCompare_c2c.bat', 'w', encoding='utf-8') as f_object:
        f_object.write(bat_string)
        f_object.write('\n')
    subprocess.call([f'{str(cloudcomparetmpfolder)}/CloudCompare_c2c.bat'])

    c2c_dist_source_filename = append_string_to_filename(source_filename, '_C2C_DIST', '.bin')
    mesh_source_filename = remove_string_from_filename(source_filename, '_SAMPLED_POINTS', '.obj')

    # Interpolate the c2c_dist point cloud back onto the mesh vertices and save the vertices with the scalar field
    filename_sf_interp_log = append_string_to_filename(source_filename, '_CC_c2c_sf_interp', '.log')
    bat_string = f'"{str(cloudcomparepath)}" -SILENT -NO_TIMESTAMP -AUTO_SAVE OFF ' \
                 f'-o "{str(c2c_dist_source_filename)}" ' \
                 f'-o "{str(mesh_source_filename)}" ' \
                 f'-LOG_FILE "{str(filename_sf_interp_log)}" ' \
                 f'-EXTRACT_VERTICES -SF_INTERP LAST ' \
                 f'-C_EXPORT_FMT PLY -PLY_EXPORT_FMT ASCII -SAVE_CLOUDS ' \
                 f''

    with open(f'{str(cloudcomparetmpfolder)}/CloudCompare_interp.bat', 'w', encoding='utf-8') as f_object:
        f_object.write(bat_string)
        f_object.write('\n')
    subprocess.call([f'{str(cloudcomparetmpfolder)}/CloudCompare_interp.bat'])

    # Load mesh vertices with scalar_c2c_absolute_distances
    mesh_vertices_source_filename = Path(str(mesh_source_filename.with_suffix("")) + '.vertices.ply')
    mesh_vertices_source = trimesh.load(mesh_vertices_source_filename, process=False, maintain_order=True)
    scalar_c2c_absolute_distances = \
        mesh_vertices_source.metadata['_ply_raw']['vertex']['data']['scalar_C2C_absolute_distances'].squeeze()

    # Normalize the scale
    max_scalar = np.max(scalar_c2c_absolute_distances)
    min_scalar = np.min(scalar_c2c_absolute_distances)
    scalar_c2c_absolute_distances = (scalar_c2c_absolute_distances - min_scalar) / (max_scalar - min_scalar)
    # max_scalar2 = np.max(scalar_c2c_absolute_distances)
    # min_scalar2 = np.min(scalar_c2c_absolute_distances)

    # Load the mesh
    mesh_source = trimesh.load(mesh_source_filename, process=False, maintain_order=True)

    my_cmap = cc_bluegreenyellowred_colormap()

    # Set the vertex colors according to c2c_dist using the cc colormap
    mesh_source.visual.vertex_colors = my_cmap(scalar_c2c_absolute_distances)

    mesh_source_filename_saved = append_string_to_filename(mesh_source_filename, '_C2C_DIST_SURFACE', '.ply')
    # Save the mesh to a file in ply format
    # trimesh.exchange.export.export_mesh(mesh_source, mesh_source_filename_saved)
    trimesh.base.export_mesh(mesh_source, mesh_source_filename_saved)

    return get_mean_stdev_from_cc_log(cc_log_filename)


def save_cloud_compare_sampled_point_cloud(output_filename: Path,
                                           cloudcomparepath: Path = Path("C:/Program Files/CloudCompare/CloudCompare"),
                                           cloudcomparetmpfolder: Path = Path("D:/tmp/star")) -> None:
    """
    Saves a point cloud of the mesh with 200000 points

    :param output_filename: point cloud output filename
    :param cloudcomparepath: path to cloud compare installation
    :param cloudcomparetmpfolder: path to a tmp folder for cloud compare batch files
    """
    if not cloudcomparetmpfolder.is_dir():
        cloudcomparetmpfolder.mkdir(parents=True)

    output_cc_log_filename = append_string_to_filename(output_filename, '_CC_c2c', '.log')
    bat_string = f'"{str(cloudcomparepath)}" -SILENT -NO_TIMESTAMP -LOG_FILE ' \
                 f'"{output_cc_log_filename}" -o "{output_filename}" -SAMPLE_MESH POINTS 200000"'
    with open(f'{str(cloudcomparetmpfolder)}/CloudCompare_sample_points.bat', 'w', encoding='utf-8') as f_object:
        f_object.write(bat_string)
        f_object.write('\n')
    subprocess.call([f'{str(cloudcomparetmpfolder)}/CloudCompare_sample_points.bat'])


def find_smpl_vertice_labels(filename: str = 'smpl_vert_segmentation.json') -> Tuple[List, List]:
    """
    Load a json file containing the list of vertices within each body part
    smpl_vert_segmentation.json file is part of the meshcapade wiki: https://github.com/Meshcapade/wiki

    :param filename: the name of the json file with the vertices' body position
    :return: a list of the vertices within the head and hands, and within the head, hands and feet of a STAR body model
    """

    f = open(filename)
    data = json.load(f)
    right_hand = data['rightHand']
    right_hand_index1 = data['rightHandIndex1']
    left_hand = data['leftHand']
    left_hand_index1 = data['leftHandIndex1']
    head = data['head']
    neck = data['neck']
    left_foot = data['leftFoot']
    left_toe_base = data['leftToeBase']
    right_foot = data['rightFoot']
    right_toe_base = data['rightToeBase']

    head_hands_index = right_hand + right_hand_index1 + left_hand + left_hand_index1 + head + neck
    head_hands_feet_index = head_hands_index + left_foot + left_toe_base + right_foot + right_toe_base

    return head_hands_index, head_hands_feet_index


def save_star_mesh_out_and_target_scan_with_no_hand_head_feet(scan_type: ScanType,
                                                              star_mesh_out_filename: Path,
                                                              star_mesh_out_no_hh_filename: Path,
                                                              star_mesh_out_no_hhf_filename: Path,
                                                              star_mesh_out_trimesh: Trimesh,
                                                              target_scan_no_hh_filename: Path,
                                                              target_scan_no_hhf_filename: Path,
                                                              target_scan_trimesh: Trimesh) -> None:
    """
    Save off the fitted star mesh and the target scan with no hands or head, and also with no hands, head or feet
    This allows for better comparison between the fitted STAR body model and the target scan without the hands, head or
    feet where significant error can occur and which we aren't as interested in

    :param scan_type: The scan_type of the scan
    :param star_mesh_out_filename: the filename of the fitted STAR body model
    :param star_mesh_out_no_hh_filename: the filename of the fitted STAR body model with no hands or head
    :param star_mesh_out_no_hhf_filename: the filename of the fitted STAR body model with no hands, head or feet
    :param star_mesh_out_trimesh: the fitted STAR body model
    :param target_scan_no_hh_filename:  the filename of the target scan with no hands or head
    :param target_scan_no_hhf_filename:  the filename of the target scan with no hands, head or feet
    :param target_scan_trimesh: the target scan
    """
    star_mesh_out_trimesh_original = star_mesh_out_trimesh.copy()

    save_trimesh(star_mesh_out_trimesh, scan_type, star_mesh_out_filename)

    # Remove head, hands and feet from star model
    v1, f1, v2, f2 = remove_head_and_hands_from_star(star_mesh_out_trimesh_original)
    star_mesh_out_no_hh_trimesh = Trimesh(v1, f1, face_colors=trimesh_color_light_grey,
                                          vertex_normals=None)
    star_mesh_out_no_hh_trimesh.remove_unreferenced_vertices()
    star_mesh_out_no_hhf_trimesh = Trimesh(v2, f2, face_colors=trimesh_color_light_grey)
    star_mesh_out_no_hhf_trimesh.remove_unreferenced_vertices()
    save_trimesh(star_mesh_out_no_hh_trimesh, scan_type, star_mesh_out_no_hh_filename)
    save_trimesh(star_mesh_out_no_hhf_trimesh, scan_type, star_mesh_out_no_hhf_filename)

    # Remove head, hands, feet from original scan
    v1, f1, v2, f2 = remove_head_and_hands_from_scan(target_scan_trimesh, star_mesh_out_trimesh_original)
    scan_mesh_out_no_hh_trimesh = None
    scan_mesh_out_no_hhf_trimesh = None
    if isinstance(target_scan_trimesh, Trimesh):
        scan_mesh_out_no_hh_trimesh = Trimesh(v1, f1,
                                              face_colors=trimesh_color_light_grey)
        scan_mesh_out_no_hh_trimesh.remove_unreferenced_vertices()
        scan_mesh_out_no_hhf_trimesh = Trimesh(v2, f2,
                                               face_colors=trimesh_color_light_grey)
        scan_mesh_out_no_hhf_trimesh.remove_unreferenced_vertices()
    elif isinstance(target_scan_trimesh, PointCloud):
        scan_mesh_out_no_hh_trimesh = PointCloud(v1, face_colors=trimesh_color_light_grey)
        scan_mesh_out_no_hhf_trimesh = PointCloud(v2, face_colors=trimesh_color_light_grey)
    save_trimesh(scan_mesh_out_no_hh_trimesh, scan_type, target_scan_no_hh_filename)
    save_trimesh(scan_mesh_out_no_hhf_trimesh, scan_type, target_scan_no_hhf_filename)


def remove_head_and_hands_from_star(mesh_target_trimesh: Trimesh) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove the head and hands from the STAR body model

    :param mesh_target_trimesh: STAR body scan to remove hands and head from

    :return: A TrackedArray for the verts and faces both without hands and head and without hands, head and feet
    """
    head_hands_index, head_hands_feet_index = find_smpl_vertice_labels()

    faces = np.array(mesh_target_trimesh.faces)
    verts = np.array(mesh_target_trimesh.vertices)

    # Test if any face contains vertices that are in the head_hands_index
    # Will get a boolean array
    faces_in_hh_index = np.any(np.isin(faces, head_hands_index), axis=1)
    # Delete any faces where faces_in_hh_index is true
    faces_without_hh = np.delete(faces, faces_in_hh_index, axis=0)
    faces_in_hhf_index = np.any(np.isin(faces, head_hands_feet_index), axis=1)
    faces_without_hhf = np.delete(faces, faces_in_hhf_index, axis=0)

    # return verts_without_hh, faces_without_hh, verts_without_hhf, faces_without_hhf
    return verts, faces_without_hh, verts, faces_without_hhf


def remove_head_and_hands_from_scan(mesh_target_trimesh: Union[Trimesh, PointCloud], mesh_out: Trimesh) -> \
        Tuple[np.ndarray, Union[None, np.ndarray], np.ndarray, Union[None, np.ndarray]]:
    """
    Remove the head and hands from a scan

    :param mesh_target_trimesh: STAR body model
    :param mesh_out: scan to remove hands and head from

    :return: A TrackedArray for the verts and faces both without hands and head and without hands, head and feet
    """

    head_hands_index, head_hands_feet_index = find_smpl_vertice_labels()

    vertices_1 = np.array(mesh_target_trimesh.vertices)
    vertices_2 = np.array(mesh_out.vertices)

    # k nearest neighbour search
    # tree1 = KDTree(vertices_1) # not used
    tree2 = KDTree(vertices_2)
    d12, d12_index = tree2.query(vertices_1, k=1)

    scan_vertex_in_star_head_or_hands = np.isin(d12_index, head_hands_index)
    scan_range = np.arange(0, len(vertices_1))
    scan_vertex_in_star_head_or_hands_index = scan_range[scan_vertex_in_star_head_or_hands]
    # scan_vertex_in_star_head_or_hands_or_feet = np.isin(d12_index, head_hands_feet_index)

    scan_vertex_in_star_head_or_hands_or_feet = np.isin(d12_index, head_hands_feet_index)
    scan_vertex_in_star_head_or_hands_or_feet_index = scan_range[scan_vertex_in_star_head_or_hands_or_feet]
    # scan_vertex_in_star_head_or_hands_or_feet = np.isin(d12_index, head_hands_feet_index)

    if isinstance(mesh_target_trimesh, Trimesh):
        faces = np.array(mesh_target_trimesh.faces)

        faces_index_no_hh = np.any(np.isin(faces, scan_vertex_in_star_head_or_hands_index), axis=1)
        faces_no_hh = np.delete(faces, faces_index_no_hh, axis=0)

        faces_index_no_hhf = np.any(np.isin(faces, scan_vertex_in_star_head_or_hands_or_feet_index), axis=1)
        faces_no_hhf = np.delete(faces, faces_index_no_hhf, axis=0)

        return vertices_1, faces_no_hh, vertices_1, faces_no_hhf
    elif isinstance(mesh_target_trimesh, PointCloud):
        vertices_1_no_hh = np.delete(vertices_1, scan_vertex_in_star_head_or_hands_index, axis=0)
        vertices_1_no_hhf = np.delete(vertices_1, scan_vertex_in_star_head_or_hands_or_feet_index, axis=0)

        return vertices_1_no_hh, None, vertices_1_no_hhf, None


def text_3d(text: str,
            pos: Tuple[float, float, float],
            direction: Tuple[float, float, float] = None,
            degree: float = 0.0,
            density: int = 10,
            font: str = 'arial.ttf',
            font_size: int = 16):
    """
    Generate a 3D text point cloud used for visualization.
    From https://github.com/isl-org/Open3D/issues/2

    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param density:
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geometry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def cc_bluegreenyellowred_colormap() -> LinearSegmentedColormap:
    """
    Create a cloud compare bluegreenyellowred colormap

    :return: A colormap as a LinearSegmentedColormap object
    """
    # Create a matplotlib colormap with the default cloudcompare bluegreenyellowred colormap
    colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    positions = [0, 0.333333333333, 0.666666666667, 1.0]

    # Create a colormap using LinearSegmentedColormap
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', list(zip(positions, colors)))
    return my_cmap
