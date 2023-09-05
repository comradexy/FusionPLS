import argparse
import os
import open3d as o3d
import numpy as np
import yaml
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, pts_path, labels_path, cfg_path):
        with open(cfg_path, "r") as stream:
            cfg = yaml.safe_load(stream)
        self.cfg = cfg
        self.pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 7)
        self.labels = np.fromfile(labels_path, dtype=np.int32).reshape(-1)
        assert self.pts.shape[0] == self.labels.shape[0], \
            "Number of points and labels must be equal"
        self.cm = self.cfg["color_map"]
        self.cml = self.cfg["color_map_learning"]
        self.sem_labels = self.labels & 0xFFFF
        self.ins_labels = self.labels >> 16

    def vis_pcd(self, size=2.0, background=(0, 0, 0), mode="sem", name="Open3D"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts[:, :3])
        if mode == "rgb":
            rgb = np.array(self.pts[:, 4:7])
            pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
        elif mode == "intensity":
            pcd.colors = o3d.utility.Vector3dVector(self.pts[:, 3:4])
        elif mode == "sem":
            sem_colors = np.array([self.cm[i] for i in self.sem_labels])
            pcd.colors = o3d.utility.Vector3dVector(sem_colors.reshape(-1, 3) / 255.0)
        elif mode == "ins":
            ins_lab = np.copy(self.ins_labels)
            sem_lab = np.copy(self.sem_labels)
            sem_lab[ins_lab == 0] = 0
            range_data = np.copy(ins_lab)
            viridis_range = ((range_data - range_data.min()) /
                             (range_data.max() - range_data.min()) *
                             255).astype(np.uint8)
            viridis_map = self.get_mpl_colormap("viridis")
            viridis_colors = viridis_map[viridis_range]
            # replace void label with [50, 50, 50] color
            viridis_colors[self.ins_labels == 0] = np.array([50, 50, 50]) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(viridis_colors.reshape(-1, 3))

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=720, height=720, window_name=name)
        vis.get_render_option().background_color = background
        vis.get_render_option().point_size = size
        vis.add_geometry(pcd)
        vis.run()

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--pts', '-p', type=str, help='points path')
parser.add_argument('--lab', '-l', type=str, help='labels path')
parser.add_argument('--cfg', '-c', type=str, help='config path')
parser.add_argument('--mode', '-m', type=str, default='sem', help='visualize mode')
parser.add_argument('--name', '-n', type=str, default='Open3D', help='window name')

if __name__ == '__main__':
    args = parser.parse_args()
    vis = Visualizer(args.pts, args.lab, args.cfg)
    vis.vis_pcd(mode=args.mode, name=args.name)
    # print(np.unique(vis.ins_labels))
