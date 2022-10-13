from unittest.mock import patch
import laspy
import numpy as np
import open3d as o3d
import os


path = os.path.join(os.getcwd(), "output/TORONTO3D_pointcloud_1.laz")
print(path)

las = laspy.read(path)
point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
print(point_data)

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([geom])




