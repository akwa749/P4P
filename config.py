import os
import pathlib

import numpy as np
import open3d as o3d
import pandas as pd

import bounding_box as bx


class PreLoad:
    def __init__(self) -> None:
        self.pcd = None

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
