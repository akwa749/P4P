import os
import pathlib

import numpy as np
import open3d as o3d
import pandas as pd


class PreLoad:
    def rename(self, dir_path="./data"):
        # renames to tmp name
        for dir, subdirs, files in os.walk(dir_path):
            for i, f in enumerate(files, 1):
                _, ext = os.path.splitext(f)
                temp_file_name = f"temp_{str(i).zfill(3)}{ext}"
                os.rename(os.path.join(dir, f), os.path.join(dir, temp_file_name))

        # renames to new name
        for dir, subdirs, files in os.walk(dir_path):
            for i, f in enumerate(files, 1):
                _, ext = os.path.splitext(f)
                new_file_name = f"P{str(i).zfill(3)}{ext}"
                os.rename(os.path.join(dir, f), os.path.join(dir, new_file_name))

    def load(self, path: str) -> o3d.cpu.pybind.geometry.PointCloud:
        """
        Loads a ply file provided a valid path with robust error checking

        Checks file path exists and is a ply file that is not empty

        Args:
            path: string path to a .ply file

        Returns:
            pcd: non-empty point cloud data object
        """

        # checks if file exists
        if not pathlib.Path(path).exists():
            raise FileNotFoundError("File not found.")

        # checking correct file format
        if pathlib.Path(path).suffix != ".ply":
            raise ValueError(f"Expected a .ply file, got {pathlib.Path(path).suffix}")

        # check if file is not empty
        if pathlib.Path(path).stat().st_size == 0:
            raise ValueError("File is empty.")

        pcd = o3d.io.read_point_cloud(path)

        # checks pt cloud is not empty
        if not np.asarray(pcd.points).size:
            raise ValueError("Point cloud has no points.")

        # pts and colors must match (not too important, but mismatches may lead to incorrect results when vis or analysing)
        if np.asarray(pcd.points).shape[0] != np.asarray(pcd.colors).shape[0]:
            raise ValueError("Point cloud has mismatch between points and colors.")

        return pcd  # should pass all error cases
