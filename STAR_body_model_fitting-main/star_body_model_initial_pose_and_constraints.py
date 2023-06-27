"""
star_body_model_initial_pose_and_constraints.py
by Conrad Werkhoven, Auckland Bioengineering Institute

https://github.com/ConradW01/STAR_body_model_fitting

Set the initial pose, body shape and translation for the STAR body model

"""
from typing import Tuple

import numpy as np

from star_body_model_utils import deg2rad, ScanType


# Note that for the SMPL/STAR model the pelvis is joint 0 in the list of pose joints
# For the SMPLX model the pelvis (joint 0) was called the global_orient and was not included in the pose joints
# So the index into the pose joint list is 3 larger for the SMPL/STAR model compared to the SMPLX model


def initial_pose(star_initial_transl: np.ndarray,
                 palm_orientation: str,
                 run_number: int = None,
                 scan_type: ScanType = None,
                 beta: float = None) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the initial pose, body shape and translation of the STAR model

    :param star_initial_transl: the initial translation to move the STAR body model over the initial scan
    :param palm_orientation: palm orientation: either 'front' or 'side'
    :param run_number: the number of the run
    :param scan_type: the type of scan, e.g. 3d, lidar or zozo
    :param beta: the value of the first principal component based on the height of the scan
    :return: combined_parameters - the pose, shape and translation.
     combined_parameters_lower_limit - the pose, shape and translation lower limit.
     combined_parameters_upper_limit - the pose, shape and translation upper limit.

    """
    init_pose = np.zeros(72, dtype=np.float32)
    pose_lower_limit = np.ones(72, dtype=np.float32) * -180 * deg2rad
    pose_upper_limit = np.ones(72, dtype=np.float32) * 180 * deg2rad

    # 7 'left_ankle',
    # 8 'right_ankle',
    init_pose[21:27] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])
    pose_lower_limit[21:27] = np.multiply(deg2rad, [0, -180, 0, 0, -180, 0])
    pose_upper_limit[21:27] = np.multiply(deg2rad, [0, 180, 0, 0, 180, 0])
    # 10 'left_foot',
    # 11 'right_foot',
    init_pose[30:36] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])
    pose_lower_limit[30:36] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])
    pose_upper_limit[30:36] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])
    # 12 'neck',
    init_pose[36:39] = np.multiply(deg2rad, [0, 0, 0])
    pose_lower_limit[36:39] = np.multiply(deg2rad, [-5, -5, -5])
    pose_upper_limit[36:39] = np.multiply(deg2rad, [5, 5, 5])
    # 15 'head',
    init_pose[45:48] = np.multiply(deg2rad, [0, 0, 0])
    pose_lower_limit[45:48] = np.multiply(deg2rad, [-5, -5, -5])
    pose_upper_limit[45:48] = np.multiply(deg2rad, [5, 5, 5])
    # # 16 'left_shoulder',
    # # 17 'right_shoulder',
    init_pose[48:54] = np.multiply(deg2rad, [0, 0, -70, 0, 0, 70])
    # # 20 'left_wrist',
    # # 21 'right_wrist',
    if palm_orientation == 'side':
        init_pose[60:66] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])
        pose_lower_limit[60:66] = np.multiply(deg2rad, [-5, -5, -5, -5, -5, -5])
        pose_upper_limit[60:66] = np.multiply(deg2rad, [5, 5, 5, 5, 5, 5])
    elif palm_orientation == 'front':
        init_pose[60:66] = np.multiply(deg2rad, [-90, 0, 0, -90, 0, 0])  # Palms to the front
        pose_lower_limit[60:66] = np.multiply(deg2rad, [-95, -5, -5, -95, -5, -5])
        pose_upper_limit[60:66] = np.multiply(deg2rad, [-85, 5, 5, -85, 5, 5])

    # # 22 'left_hand'
    # # 23 'right_hand'
    init_pose[66:72] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])
    pose_lower_limit[66:72] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])
    pose_upper_limit[66:72] = np.multiply(deg2rad, [0, 0, 0, 0, 0, 0])

    # SHAPE PARAMETERS
    init_betas = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    betas_lower_limit = np.multiply(-20, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    betas_upper_limit = np.multiply(20, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Special cases
    if run_number == 2 and scan_type == ScanType.lidar:
        # # 16 'left_shoulder',
        # # 17 'right_shoulder',
        init_pose[48:54] = np.multiply(deg2rad, [0, 0, -73, 0, 0, 73])
    elif run_number == 11 and scan_type == ScanType.td:
        # # 16 'left_shoulder',
        # # 17 'right_shoulder',
        init_pose[48:54] = np.multiply(deg2rad, [0, 0, -80, 0, 0, 80])
        init_betas = np.array([-0.12, -1.93, -2.96, -1.36, 0.49, 0.04, 0.55, -3.26, 3.41, 1.60])
    elif run_number == 20 and scan_type == ScanType.lidar:
        # global orientation
        init_pose[0:3] = np.multiply(deg2rad, [0, -90, 0])
    elif run_number == 22 and scan_type == ScanType.lidar:
        # 1 'left_hip',
        # 2 'right_hip',
        init_pose[3:9] = np.multiply(deg2rad, [0, 0, -3, 0, 0, 2])
        pose_lower_limit[3:9] = np.multiply(deg2rad, [0, -5, 0, 0, -5, 0])
        pose_upper_limit[3:9] = np.multiply(deg2rad, [0, 5, 0, 0, 5, 0])
        # left knee
        # right knee
        pose_lower_limit[12:18] = np.multiply(deg2rad, [0, -5, 0, 0, -5, 0])
        pose_upper_limit[12:18] = np.multiply(deg2rad, [0, 5, 0, 0, 5, 0])
        # left ankle
        # right ankle
        pose_lower_limit[21:27] = np.multiply(deg2rad, [0, -5, 0, 0, -5, 0])
        pose_upper_limit[21:27] = np.multiply(deg2rad, [0, 5, 0, 0, 5, 0])
        init_betas = np.array([-0.87, -2.21, 1.99, -1.96, -0.49, -2.56, -2.97, 1.94, -1.41, -1.33])
    elif run_number == 24 and scan_type == ScanType.lidar:
        # 16 'left_shoulder',
        # 17 'right_shoulder',
        init_pose[48:54] = np.multiply(deg2rad, [2, 2, -56, 2, -2, 56])  # arms further from body
        # left collar only
        pose_lower_limit[41:42] = np.multiply(deg2rad, [0])
        pose_upper_limit[41:42] = np.multiply(deg2rad, [0])
    elif run_number == 30 and scan_type == ScanType.td:
        pass

    # BODY TRANSLATION PARAMETERS
    init_body_translation = star_initial_transl
    body_translation_lower_limit = np.multiply(-3, [1, 1, 1])
    body_translation_upper_limit = np.multiply(3, [1, 1, 1])

    # Combined Translation and Global Orientation and Shape and Pose and parameters
    combined_parameters = np.zeros(85, dtype=np.float32)
    combined_parameters_lower_limit = np.zeros(85, dtype=np.float32)
    combined_parameters_upper_limit = np.zeros(85, dtype=np.float32)

    # Make one big numpy array
    combined_parameters[:72] = init_pose
    combined_parameters[72:82] = init_betas
    combined_parameters[82:85] = init_body_translation

    combined_parameters_lower_limit[:72] = pose_lower_limit
    combined_parameters_lower_limit[72:82] = betas_lower_limit
    combined_parameters_lower_limit[82:85] = body_translation_lower_limit

    combined_parameters_upper_limit[:72] = pose_upper_limit
    combined_parameters_upper_limit[72:82] = betas_upper_limit
    combined_parameters_upper_limit[82:85] = body_translation_upper_limit

    # Expand dims and convert to pytorch Tensor
    combined_parameters = np.expand_dims(combined_parameters, axis=0)

    if beta:
        combined_parameters[0, 72] = beta  # Set initial height

    return combined_parameters, combined_parameters_lower_limit, combined_parameters_upper_limit
