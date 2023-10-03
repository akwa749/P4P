import copy

import ipywidgets as widgets
import numpy as np
import open3d as o3d
from IPython.display import display
from ipywidgets import Button, FloatSlider, FloatText, interactive
from ipywidgets import jslink as link


class BoundingBox:
    def __init__(self, pcd):
        self.pcd = pcd
        self.final_pcd = None  # holds final state of point cloud
        self.create_widgets()

    def create_widgets(self):
        pcd_tmp = np.asarray(self.pcd.points)
        min_x, min_y, min_z = pcd_tmp.min(axis=0)
        max_x, max_y, max_z = pcd_tmp.max(axis=0)

        self.save_button = Button(description="Save point cloud")
        self.save_button.on_click(self.save_pcd)

        ## SLIDER X
        self.slider_x = FloatSlider(
            min=min_x, max=max_x, step=0.001, value=-0.8, description="x"
        )
        self.text_x = FloatText(value=min_x, description="x")

        ## SLIDER X NEGATIVE VERSION
        self.slider_x_dup = FloatSlider(
            min=min_x, max=max_x, step=0.001, value=0.8, description="x_dup"
        )
        self.text_x_dup = FloatText(value=max_x, description="x_dup")

        ## SLIDER Y
        self.slider_y = FloatSlider(
            min=min_y, max=max_y, step=0.001, value=0.8, description="y"
        )
        self.text_y = FloatText(value=max_y, description="y")

        ## SLIDER Y NEGATIVE VERSION
        self.slider_y_dup = FloatSlider(
            min=min_y, max=max_y, step=0.001, value=0.8, description="y_dup"
        )
        self.text_y_dup = FloatText(value=max_y, description="y_dup")

        self.slider_z = FloatSlider(
            min=min_z, max=max_z, step=0.001, value=0.03, description="z"
        )
        self.text_z = FloatText(value=min_z, description="z")
        link((self.slider_z, "value"), (self.text_z, "value"))

        # Link sliders to text boxes
        link((self.slider_x, "value"), (self.text_x, "value"))
        link((self.slider_x_dup, "value"), (self.text_x_dup, "value"))
        link((self.slider_y, "value"), (self.text_y, "value"))
        link((self.slider_y_dup, "value"), (self.text_y_dup, "value"))
        link((self.slider_z, "value"), (self.text_z, "value"))

    def create_grid_on_plane(self, thresh, axis, color):
        lines = []
        colors = []
        for i in np.linspace(-1, 1, 10):  # Adjust these values as per your requirements
            # vertical lines
            start = (
                [i, thresh, -1]
                if axis == "y"
                else ([thresh, i, -1] if axis == "x" else [-1, i, thresh])
            )
            end = (
                [i, thresh, 1]
                if axis == "y"
                else ([thresh, i, 1] if axis == "x" else [1, i, thresh])
            )
            lines.append([start, end])
            colors.append(color)
            # horizontal lines
            start = (
                [-1, thresh, i]
                if axis == "y"
                else ([thresh, -1, i] if axis == "x" else [i, -1, thresh])
            )
            end = (
                [1, thresh, i]
                if axis == "y"
                else ([thresh, 1, i] if axis == "x" else [i, 1, thresh])
            )
            lines.append([start, end])
            colors.append(color)

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3)),
            lines=o3d.utility.Vector2iVector(
                np.array([[i, i + 1] for i in range(0, len(lines) * 2, 2)])
            ),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def view_and_adjust_threshold(
        self, thresh_x, thresh_x_dup, thresh_y=None, thresh_y_dup=None, thresh_z=None
    ):
        pcd_tmp = np.asarray(self.pcd.points)
        pcd_tmpc = np.asarray(self.pcd.colors)

        indices = (
            ((pcd_tmp[:, 0] > thresh_x) & (pcd_tmp[:, 0] < thresh_x_dup))
            & ((pcd_tmp[:, 1] > thresh_y) & (pcd_tmp[:, 1] < thresh_y_dup))
            & (pcd_tmp[:, 2] > thresh_z)
        )
        pcd_tmp_tmp = pcd_tmp[indices]
        pcd_tmp_tmpc = pcd_tmpc[indices]

        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(pcd_tmp_tmp)
        filtered.colors = o3d.utility.Vector3dVector(pcd_tmp_tmpc)

        self.final_pcd = filtered

        plane_x = self.create_grid_on_plane(
            thresh_x, "x", [1, 0, 0]
        )  # red color for x plane
        plane_y = self.create_grid_on_plane(
            thresh_y, "y", [0, 1, 0]
        )  # green color for y plane
        plane_z = self.create_grid_on_plane(
            thresh_z, "z", [0, 0, 1]
        )  # blue color for z plane

        plane_x_dup = self.create_grid_on_plane(thresh_x_dup, "x", [1, 0, 0])
        plane_y_dup = self.create_grid_on_plane(thresh_y_dup, "y", [0, 1, 0])

        centroid = np.mean(np.asarray(self.pcd.points), axis=0)
        centroid_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=centroid
        )

        ## visualise origin pt
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [
                self.final_pcd,
                centroid_frame,
                origin,
                plane_x,
                plane_y,
                plane_z,
                plane_x_dup,
                plane_y_dup,
            ]
        )

        pcd = copy.deepcopy(self.final_pcd)

    def start_interaction(self):
        interactive(
            self.view_and_adjust_threshold,
            thresh_x=self.slider_x,
            thresh_x_dup=self.slider_x_dup,
            thresh_y=self.slider_y,
            thresh_y_dup=self.slider_y_dup,
            thresh_z=self.slider_z,
        )

        display(
            self.text_x,
            self.text_x_dup,
            self.text_y,
            self.text_y_dup,
            self.text_z,
            self.save_button,
        )

    def save_pcd(self):
        if self.final_pcd is not None:
            o3d.io.write_point_cloud("final.ply", self.final_pcd)
            print("Point cloud saved!")
        else:
            print("No point cloud to save.")
