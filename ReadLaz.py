import laspy
import numpy as np
import open3d as o3d

las = laspy.read("/mnt/Lidar Related/Data Samples/4800E_54550N.las")
point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
print(point_data)

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([geom])