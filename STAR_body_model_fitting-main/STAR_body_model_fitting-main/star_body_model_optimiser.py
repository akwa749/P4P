"""
star_body_model_optimiser.py
by Conrad Werkhoven, Auckland Bioengineering Institute

https://github.com/ConradW01/STAR_body_model_fitting

Fit a STAR body model to a mesh.

"""
import csv
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import open3d as o3d
import pytorch3d
import torch
from pytorch3d import loss
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from star.pytorch.star import STAR
from torch import Tensor
from trimesh import Trimesh, PointCloud

from star_body_model_utils import create_open3d_mesh_from_verts_and_faces, o3d_color_light_grey, rad2deg, BLACK, \
    RED, \
    STAR_JOINT_NAMES, o3d_color_light_red, trimesh_color_light_grey, ScanType, trimesh_to_torch, trimesh_to_open3d

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class StarBodyModelOptimiser:

    def __init__(self,
                 target_scan_trimesh: Union[Trimesh, PointCloud],
                 target_scan_name: str,
                 scan_type: ScanType,
                 gender: str,
                 initial_combined_parameters_numpy: np.ndarray,
                 initial_combined_parameters_lower_limit_numpy: np.ndarray,
                 initial_combined_parameters_upper_limit_numpy: np.ndarray,
                 output_folder: Path,
                 max_iters: int = 401,
                 n_points_in_pointcloud: int = 200000,
                 visualize: bool = True,
                 constrain: bool = True,
                 save_images: bool = True):
        """
        :param target_scan_trimesh: the target scan, either a trimesh or PointCloud
        :param target_scan_name: the name of the target scan
        :param gender: either 'male' or 'female'
        :param initial_combined_parameters_numpy: a numpy array giving the initial pose and shape
        :param initial_combined_parameters_lower_limit_numpy:
         a numpy array giving the initial pose and shape lower limits
        :param initial_combined_parameters_upper_limit_numpy:
         a numpy array giving the initial pose and shape upper limits
        :param output_folder: the output folder path
        :param max_iters: the number of steps in the optimisation
        :param n_points_in_pointcloud: number of points in the pointcloud sampling of the target mesh
        :param visualize: if True draw results to screen
        :param constrain: if True then constrain optimisation based on the lower and upper limits above,
         else place no constraints on optimisation
        :param save_images: save a png of the visualization to disk
        """
        self.target_scan_name = target_scan_name
        self.gender = gender
        self.initial_combined_parameters_numpy = initial_combined_parameters_numpy
        self.initial_combined_parameters_lower_limit_numpy = initial_combined_parameters_lower_limit_numpy
        self.initial_combined_parameters_upper_limit_numpy = initial_combined_parameters_upper_limit_numpy
        self.output_folder = output_folder
        self.max_iters = max_iters
        self.n_points_in_pointcloud = n_points_in_pointcloud
        self.visualize = visualize  # Visualize the meshes/point clouds
        self.constrain = constrain  # Add limits to pose and shape in optimization
        self.save_images = save_images  # save image for each step - used to make a movie if wanted
        self.scan_type = scan_type

        # Load mesh/pointcloud as open3d and pytorch objects
        # Pytorch is used for optimisation
        # Opend3d is used for displaying on screen
        # Trimesh is used for loading and saving
        target_pointcloud_torch = None
        # self.target_scan_o3d = o3d.geometry.TriangleMesh()
        if isinstance(target_scan_trimesh, Trimesh):
            target_mesh_o3d = target_scan_trimesh.as_open3d
            target_mesh_o3d.compute_vertex_normals()
            target_mesh_o3d.paint_uniform_color(o3d_color_light_red)
            self.target_scan_o3d = target_mesh_o3d
            target_mesh_torch = trimesh_to_torch(target_scan_trimesh)
            target_tensor_torch = \
                sample_points_from_meshes(target_mesh_torch, self.n_points_in_pointcloud)
            target_pointcloud_torch = pytorch3d.structures.Pointclouds(
                points=[target_tensor_torch.squeeze()])
        elif isinstance(target_scan_trimesh, PointCloud):
            # pointcloud_target_o3d = target_scan_trimesh.as_open3d
            pointcloud_target_o3d = trimesh_to_open3d(target_scan_trimesh)
            self.target_scan_o3d = pointcloud_target_o3d
            target_pointcloud_torch = trimesh_to_torch(target_scan_trimesh)

        # Convert target to pointcloud to use in fitting
        self.target_pointcloud_torch = target_pointcloud_torch.to(device)

        self.output_of_star_body_model_to_adjust = None  # Output of STAR body model fitting
        self.output_of_star_body_model_zero_pose = None  # Output of zero-pose STAR body model fitting

        self.image_folder = Path(output_folder, f'{target_scan_name}_{"images"}')  # the name of image folder
        if not self.image_folder.is_dir():
            self.image_folder.mkdir(parents=True)

        # Loss and constraint filenames
        self.loss_csv_filename = Path(output_folder, f'{target_scan_name}_loss.csv')

        # Load the STAR model to the GPU
        self.star_body_model_to_adjust = STAR(gender=self.gender).to(device)
        self.star_body_model_zero_pose = STAR(gender=self.gender).to(device)

        # Zero-pose body pose and translation arrays used in zero-pose fitting
        self.body_pose_zero = torch.FloatTensor(np.zeros((1, 72))).to(device)
        self.transl_zero = torch.FloatTensor(np.zeros((1, 3))).to(device)

    def run(self):

        # Draw to screen
        vis = None
        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=900, height=1800, left=100)

        # Move combined parameters to cuda device
        combined_parameters_torch = torch.from_numpy(self.initial_combined_parameters_numpy).to(device)
        combined_parameters_torch.requires_grad = True

        # Move constraints to cuda device
        initial_combined_parameters_lower_limit_torch = \
            torch.from_numpy(self.initial_combined_parameters_lower_limit_numpy).squeeze().to(device)
        initial_combined_parameters_upper_limit_torch = \
            torch.from_numpy(self.initial_combined_parameters_upper_limit_numpy).squeeze().to(device)

        lr = 0.01  # learning rate - essentially size of change in parameters at each step
        optimizer = torch.optim.Adam([{'params': combined_parameters_torch, 'lr': lr}])

        step = 0

        # Run optimization

        # Open a csv to save the loss at each step
        csv_writer_file = open(self.loss_csv_filename, mode='w', newline='', encoding='utf-8')
        csv_writer_loss = csv.writer(csv_writer_file, delimiter=';')

        # Initially set constraints to match pose/shape of body model. We will then relax constraints of
        # areas we wish to optimise, e.g. relax translation constraints first, then rotation, then shape and pose
        combined_parameters_lower_limit_torch = \
            torch.from_numpy(self.initial_combined_parameters_numpy).squeeze().to(device)
        combined_parameters_upper_limit_torch = \
            torch.from_numpy(self.initial_combined_parameters_numpy).squeeze().to(device)

        while step < self.max_iters:

            # Initialize optimizer
            optimizer.zero_grad()

            # Get the body pose, the betas and the translation separately for STAR model
            body_pose = combined_parameters_torch[:, :72]
            betas = combined_parameters_torch[:, 72:82]
            transl = combined_parameters_torch[:, 82:85]

            # Calculate star body model
            if step < 50:  # Don't really need these outer if statements but will leave for clarity
                # Optimise translation first for 50 steps
                if step == 0:
                    # Relax translation parameters (82:85) which is set to the initial constraints
                    # I.e. the translation parameters  are free to move within the initial constraints
                    # The other parameters stay fixed to the initial pose
                    combined_parameters_lower_limit_torch[82:85] = initial_combined_parameters_lower_limit_torch[82:85]
                    combined_parameters_upper_limit_torch[82:85] = initial_combined_parameters_upper_limit_torch[82:85]
            elif 50 <= step < 100:
                # Now optimise rotation for 50 steps
                if step == 50:
                    # Set the rotation axis constraints to the initial rotation axis constraints so that the model is
                    # now free to rotate
                    combined_parameters_lower_limit_torch[:3] = initial_combined_parameters_lower_limit_torch[:3]
                    combined_parameters_upper_limit_torch[:3] = initial_combined_parameters_upper_limit_torch[:3]
            elif 100 <= step < 125:
                # Now optimise arms and legs for 25 steps
                if step == 100:
                    # Free up constraints for arms and legs
                    combined_parameters_lower_limit_torch[np.r_[3:9, 48:54]] = \
                        initial_combined_parameters_lower_limit_torch[np.r_[3:9, 48:54]]
                    combined_parameters_upper_limit_torch[np.r_[3:9, 48:54]] = \
                        initial_combined_parameters_upper_limit_torch[np.r_[3:9, 48:54]]
            elif 125 <= step < 150:
                # Now optimise body shape for 25 steps
                if step == 125:
                    combined_parameters_lower_limit_torch[72:82] = initial_combined_parameters_lower_limit_torch[72:82]
                    combined_parameters_upper_limit_torch[72:82] = initial_combined_parameters_upper_limit_torch[72:82]
            elif 150 <= step:
                # Optimize all
                if step == 150:
                    combined_parameters_lower_limit_torch = initial_combined_parameters_lower_limit_torch
                    combined_parameters_upper_limit_torch = initial_combined_parameters_upper_limit_torch

            self.output_of_star_body_model_to_adjust = self.star_body_model_to_adjust.forward(pose=body_pose,
                                                                                              betas=betas,
                                                                                              trans=transl)

            if step == self.max_iters - 1:
                # Calculate a body model with the zero pose
                self.output_of_star_body_model_zero_pose = self.star_body_model_zero_pose.forward(
                    pose=self.body_pose_zero,
                    betas=betas,
                    trans=self.transl_zero)

            # Make pytorch mesh for optimisation
            verts_to_adjust = self.output_of_star_body_model_to_adjust[-1, :, :]
            faces_to_adjust = Tensor(self.output_of_star_body_model_to_adjust.f.astype(np.int64)).to(device)
            star_mesh_to_adjust_torch = Meshes(verts=[verts_to_adjust], faces=[faces_to_adjust])
            star_mesh_to_adjust_torch.device = device

            self.visualize_step(step, vis)

            height = float(np.max(verts_to_adjust.detach().cpu().numpy()[:, 1])) - float(
                np.min(verts_to_adjust.detach().cpu().numpy()[:, 1]))
            print('star height:', height)

            # Compute the distance between a pointcloud and a mesh within a batch.
            star_mesh_to_target_pointcloud_loss = \
                pytorch3d.loss.point_mesh_face_distance(star_mesh_to_adjust_torch, self.target_pointcloud_torch)

            # Optimization step
            # star_mesh_to_target_pointcloud_loss.backward(retain_graph=True)
            star_mesh_to_target_pointcloud_loss.backward()
            optimizer.step()

            if self.constrain:
                self.save_and_print_constraint_info(
                    combined_parameters_lower_limit_torch,
                    combined_parameters_upper_limit_torch,
                    combined_parameters_torch,
                    step)

            print('mesh:', self.target_scan_name, 'step:', step, 'loss:',
                  star_mesh_to_target_pointcloud_loss.detach().cpu().numpy())

            csv_writer_loss.writerow([f'{self.target_scan_name},{step},'
                                      f'{star_mesh_to_target_pointcloud_loss.detach().cpu().numpy()}'])

            step += 1

        # Save fitted star mesh
        verts = self.output_of_star_body_model_to_adjust[-1, :, :].detach().cpu().numpy()
        faces = self.output_of_star_body_model_to_adjust.f.astype(np.int64)
        star_mesh_out_trimesh = Trimesh(verts, faces, face_colors=trimesh_color_light_grey)

        # Save zero pose star mesh
        verts = self.output_of_star_body_model_zero_pose[-1, :, :].detach().cpu().numpy()
        faces = self.output_of_star_body_model_zero_pose.f.astype(np.int64)
        star_mesh_out_zero_pose_trimesh = Trimesh(verts, faces,
                                                  face_colors=trimesh_color_light_grey)

        return star_mesh_out_trimesh, star_mesh_out_zero_pose_trimesh

    def save_and_print_constraint_info(
            self,
            combined_parameters_lower_limit_torch: Tensor,
            combined_parameters_upper_limit_torch: Tensor,
            combined_parameters_torch: Tensor,
            step: int) -> None:
        """
        Print parameters and constraints to screen and disk

        :param combined_parameters_lower_limit_torch: a torch array giving the initial pose and shape lower limits
        :param combined_parameters_upper_limit_torch: a torch array giving the initial pose and shape upper limits
        :param combined_parameters_torch: a torch array giving the initial pose and shape
        :param step: the optimisation step number
        """

        # Use the torch clamp function to constrain the combined parameters
        combined_parameters_torch.data = torch.clamp(
            combined_parameters_torch.data,
            min=combined_parameters_lower_limit_torch,
            max=combined_parameters_upper_limit_torch)

        # Print parameters and constraints to screen
        pose_constraints_string, pose_constraints_string_black, pose_constraints_string_simple, \
            beta_constraints_string, transl_constraints_string \
            = self.print_output_data_star(combined_parameters_torch, combined_parameters_lower_limit_torch,
                                          combined_parameters_upper_limit_torch)

        # Also save constraints to file
        constraints_txt_filename = Path(self.image_folder, f'{self.target_scan_name}_parameters_step_{step}.txt')
        self.write_constraints_to_file(constraints_txt_filename, beta_constraints_string +
                                       pose_constraints_string_black + transl_constraints_string)
        constraints_simple_txt_filename = \
            Path(self.image_folder, f'{self.target_scan_name}_parameters_simple_step_{step}.txt')
        self.write_constraints_to_file(constraints_simple_txt_filename, beta_constraints_string +
                                       pose_constraints_string_simple + transl_constraints_string)

    def visualize_step(self, step: int, vis: o3d.visualization.Visualizer) -> None:
        """
        Draw the STAR fitting to screen

        :param step: the step number of the optimisation
        :param vis: an o3d.visualization.Visualizer object
        """
        if step % 1 == 0 and self.visualize:
            # Draw to screen
            vis.clear_geometries()
            # Add open3d target mesh/pointcloud
            vis.add_geometry(self.target_scan_o3d)

            # Add open3d mesh to adjust from a star body model
            star_mesh_to_adjust_o3d = create_open3d_mesh_from_verts_and_faces(verts=o3d.utility.Vector3dVector(
                self.output_of_star_body_model_to_adjust[-1, :, :].detach().cpu().numpy().squeeze()),
                faces=o3d.utility.Vector3iVector(self.output_of_star_body_model_to_adjust.f))

            # Paint it light grey
            star_mesh_to_adjust_o3d.paint_uniform_color(o3d_color_light_grey)
            vis.add_geometry(star_mesh_to_adjust_o3d)

            vis.poll_events()
            vis.update_renderer()
            if self.save_images:
                vis.capture_screen_image(str(Path(self.image_folder, f'{self.target_scan_name}_s{step}.png')))

    @staticmethod
    def write_constraints_to_file(constraints_txt_filename: Path, constraints_string: List) -> None:
        """
        Write the body shape parameters and constraints to file

        :param constraints_txt_filename: the filename path
        :param constraints_string: the text string containing the parameters and constraints
        """
        with open(constraints_txt_filename, 'w', encoding='utf-8') as f:
            for line in constraints_string:
                f.write(line)
                f.write('\n')

    @staticmethod
    def print_output_data_star(
            combined_parameters_torch: Tensor,
            combined_parameters_lower_limit_torch: Tensor,
            combined_parameters_upper_limit_torch: Tensor) -> \
            Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Create a list of strings with the parameter values and the constraint values
         to be printed to the screen and file

        :param combined_parameters_torch: the combined parameters: pose, shape translation
        :param combined_parameters_lower_limit_torch: the constraints
        :param combined_parameters_upper_limit_torch:
        :return: pose_constraints_string
        pose_constraints_string_black
        pose_constraints_string_simple
        beta_constraints_string
        transl_constraints_string
        """

        # Also calculate it in degrees
        combined_parameters_numpy = combined_parameters_torch.detach().squeeze().cpu().numpy()
        combined_parameters_in_degrees_numpy = combined_parameters_numpy.copy()
        # Convert the pose from radians to degrees
        combined_parameters_in_degrees_numpy[:72] = combined_parameters_numpy[:72] * rad2deg
        b = combined_parameters_in_degrees_numpy.tolist()
        cl = combined_parameters_lower_limit_torch.squeeze().cpu().numpy()[:72] * rad2deg
        cu = combined_parameters_upper_limit_torch.squeeze().cpu().numpy()[:72] * rad2deg
        bl = combined_parameters_lower_limit_torch.squeeze().cpu().numpy()
        bu = combined_parameters_upper_limit_torch.squeeze().cpu().numpy()

        # Print output in red if shape and pose outside limits
        lim_format = [RED if (b[i] <= cl[i] or b[i] >= cu[i]) else BLACK for i in range(0, len(b) - 13, 1)]

        pose_constraints_string = [
            f'{STAR_JOINT_NAMES[j]:15s}:( {cl[i]:4.0f} {cl[i + 1]:4.0f} {cl[i + 2]:4.0f} )( ' +
            lim_format[i] + f'{b[i]:6.2f}' + BLACK + ' ' +
            lim_format[i + 1] + f'{b[i + 1]:6.2f}' + BLACK + ' ' +
            lim_format[i + 2] + f'{b[i + 2]:6.2f}' + BLACK + ' ' +
            f' )( {cu[i]:4.0f} {cu[i + 1]:4.0f} {cu[i + 2]:4.0f} ) '
            for i, j in zip(range(0, len(b) - 13, 3), range(0, len(STAR_JOINT_NAMES), 1))]

        pose_constraints_string_black = [
            f'{STAR_JOINT_NAMES[j]:15s}:( {cl[i]:4.0f} {cl[i + 1]:4.0f} {cl[i + 2]:4.0f} )( ' +
            f'{b[i]:6.2f} {b[i + 1]:6.2f} {b[i + 2]:6.2f} ' +
            f' )( {cu[i]:4.0f} {cu[i + 1]:4.0f} {cu[i + 2]:4.0f} ) '
            for i, j in zip(range(0, len(b) - 13, 3), range(0, len(STAR_JOINT_NAMES), 1))]

        pose_constraints_string_simple = [
            f'{STAR_JOINT_NAMES[j]:15s} {cl[i]:4.0f} {cl[i + 1]:4.0f} {cl[i + 2]:4.0f} ' +
            f'{b[i]:6.2f} {b[i + 1]:6.2f} {b[i + 2]:6.2f} ' +
            f'{cu[i]:4.0f} {cu[i + 1]:4.0f} {cu[i + 2]:4.0f}'
            for i, j in zip(range(0, len(b) - 13, 3), range(0, len(STAR_JOINT_NAMES), 1))]

        beta_constraints_string = [f'beta_{i}: {bl[i + 72]:6.2f} {b[i + 72]:6.2f} {bu[i + 72]:6.2f}'
                                   for i in range(10)]

        transl_constraints_string = [
            f'translation_{i}: {bl[i + 82]:6.2f} {b[i + 82]:6.2f} {bu[i + 82]:6.2f}' for i in range(3)]

        print(*pose_constraints_string, sep='\n')
        print(*beta_constraints_string, sep='\n')
        print(*transl_constraints_string, sep='\n')

        return pose_constraints_string, pose_constraints_string_black, pose_constraints_string_simple, \
            beta_constraints_string, transl_constraints_string
