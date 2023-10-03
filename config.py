import os
import pathlib

import numpy as np
import open3d as o3d
import pandas as pd

import bounding_box as bx


class PreLoad:
    def __init__(self) -> None:
        self.pcd = None

    def rename(self, path="./data/", files=[]):
        """
        Renames files that are not subdirectory files
        """
        renamed_files = []

        # iterate through the files that are not of the same format
        for i, file in enumerate(os.listdir(path), 1):
            full_path = os.path.join(path, file)

            # finding only files ie. only .ply formatted files
            for file_to_rename in files:
                if (
                    os.path.isfile(full_path) and file == file_to_rename
                ):  # renaming only desired files
                    _, ext = os.path.splitext(file)  # extract the .ply extension
                    new_name = f"P{str(i-3).zfill(3)}{ext}"
                    new_full_path = os.path.join(path, new_name)
                    os.rename(full_path, new_full_path)
                    renamed_files.append((file, new_name))

        return renamed_files

    def load(self, path: str):
        """
        Loading in the Point Cloud Data
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

        self.pcd = o3d.io.read_point_cloud(path)

        # checks pt cloud is not empty
        if not np.asarray(self.pcd.points).size:
            raise ValueError("Point cloud has no points.")

        # pts and colors must match (not too important, but mismatches may lead to incorrect results when vis or analysing)
        if np.asarray(self.pcd.points).shape[0] != np.asarray(self.pcd.colors).shape[0]:
            raise ValueError("Point cloud has mismatch between points and colors.")

        return self.pcd  # should pass all error cases

    # def bounding_box(self):
    #     pcd_tmp = np.asarray(self.pcd.points)

    #     min_x, min_y, min_z = pcd_tmp.min(axis=0)
    #     max_x, max_y, _ = pcd_tmp.max(axis=0)

    #     bb = bx.BoundingBox(self.pcd)

    #     # Call its methods
    #     bb.view_and_adjust_threshold(
    #         thresh_x=min_x,
    #         thresh_x_dup=max_x,
    #         thresh_y=min_y,
    #         thresh_y_dup=max_y,
    #         thresh_z=min_z,
    #     )

    #     bb.start_interaction()
