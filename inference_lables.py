from enum import Enum
from pickle import FALSE
from re import sub
from trace import Trace
from turtle import color, shape
from tool import DataProcessing as DP
from helper_tool import Plot
import open3d as o3d
import numpy as np
import pandas as pd
import laspy as lp


#print("Load a ply point cloud, print it, and render it")
#pcd = o3d.io.read_point_cloud("./Dataset/SensatUrban/original_block_ply/cambridge_block_8.ply")
#o3d.visualization.draw_geometries([pcd])
#print(pcd)
#print(np.asarray(pcd.points))

DATASET = "TORONTO3D"  #"SENSATURBAN" "TORONTO3D" "SEMANTIC3D"
sampling = True


def read_points(f):
    # reads Semantic3D .txt file f into a pandas dataframe
    col_names = ['x', 'y', 'z', 'i', 'r', 'g', 'b']
    col_dtype = {'x': np.float32, 'y': np.float32, 'z': np.float32, 'i': np.int32,
                  'r': np.uint8, 'g': np.uint8, 'b': np.uint8}
    return pd.read_csv(f, names=col_names, dtype=col_dtype, delim_whitespace=True)
def read_labels(f):
    # reads Semantic3D .labels file f into a pandas dataframe
    return pd.read_csv(f, header=None)[0].values



""" 
Toronto Lables:
Unclassified 0
Ground 1
Road_markings 2
Natural 3
Building 4
Utility_line 5
Pole 6
Car 7
Fence 8 
"""

if DATASET == "TEST":
    pc_path = "../Open3D-ML/data/test/SEMANTIC3D_pointcloud.ply"
    random_sample_ratio = 3
    building_lable = 4
    eps = 3

    data = o3d.t.io.read_point_cloud(pc_path).point

    points = data["positions"].numpy()
    points = np.float32(points)
    feat = data["colors"].numpy().astype(np.float32)
    labels = np.loadtxt('../Open3D-ML/data/test/SEMANTIC3D_pointcloud.labels').astype(np.int32) + 1
    print(labels)

elif DATASET == "TORONTO3D":
    pc_path = "./Data/Toronto3D/L002.ply"
    random_sample_ratio = 5
    #4 for toronto3d infer L002, 16 for kitti infer L002
    building_lable = 4
    eps = 5
    UTM_OFFSET = [627285, 4841948, 0]
    print("start to read point cloud")

    data = o3d.t.io.read_point_cloud(pc_path).point
    #data_io = o3d.io.read_point_cloud(pc_path)
    #print(data["positions"])
    points = data["positions"].numpy()
    #points = np.asarray(data_io.points)
    points = np.float32(points)
    #feat = np.asarray(data_io.colors).astype(np.float32)
    feat = data["colors"].numpy().astype(np.float32)
    #labels = data['scalar_Label'].numpy().astype(np.int32).reshape((-1,))
    #lables are read from inferenced file
    labels = np.loadtxt('./Data/Toronto3D/L002.kitti.labels').astype(np.int32) + 1
    print(labels)

elif DATASET == "SENSATURBAN":
    pc_path = "./Data/Toronto3D/L005.ply"
    building_lable = 4
    random_sample_ratio = 3
    eps = 1

    data = o3d.t.io.read_point_cloud(pc_path).point
    points = data["positions"].numpy()
    points = np.float32(points)
    feat = data["colors"].numpy().astype(np.float32)
    #labels = data['class'].numpy().astype(np.int32).reshape((-1,))
    labels = np.loadtxt('./Data/Toronto3D/L005.ply.labels').astype(np.int32) + 1

elif DATASET == "SEMANTIC3D":
    #16 for kitti infer semantic3d, 4 for toronto3d infer semantica3d
    building_lable = 16
    random_sample_ratio = 40
    eps = 3

    pc_path = "../Open3D-ML/data/Semantic3D/bildstein_station1_xyz_intensity_rgb_ori.txt"
    label_path = "../Open3D-ML/data/Semantic3D/bildstein_station1_xyz_intensity_rgb_ori.kitti.labels"
    points = read_points(pc_path)
    print(points.shape)
    labels = read_labels(label_path) + 1
    print(labels.shape)
    pc_xyzrgb = np.vstack((points['x'], points['y'], points['z'], points['r'], points['g'], points['b'])).T
    points = pc_xyzrgb[:, 0:3]
    feat = pc_xyzrgb[:, 3:6] 
    labels = labels.astype(int)

else:

    print("not support dataset!")
    exit()


data = {'point': points, 'feat': feat, 'label': labels}

""" 
SensatUrban Lables:
0-Ground: including impervious surfaces, grass, terrain
1-Vegetation: including trees, shrubs, hedges, bushes
2-Building: including commercial / residential buildings
3-Wall: including fence, highway barriers, walls
4-Bridge: road bridges
5-Parking: parking lots
6-Rail: railroad tracks
7-Traffic Road: including main streets, highways
8-Street Furniture: including benches, poles, lights
9-Car: including cars, trucks, HGVs
10-Footpath: including walkway, alley
11-Bike: bikes / bicyclists
12-Water: rivers / water canals
"""

#xyz, rgb, labels = DP.read_ply_data("./Dataset/SensatUrban/original_block_ply/birmingham_block_7.ply", with_rgb=True)
#print(labels)

random_sample_ratio = random_sample_ratio
if sampling:
    sub_xyz, sub_rgb, sub_labels = DP.random_sub_sampling(data['point'], data['feat'], data['label'], random_sample_ratio)
else:
    sub_xyz, sub_rgb, sub_labels = data['point'], data['feat'], data['label']

print(sub_xyz.shape)

plot_colors = Plot.random_colors(20, seed=2)

#Plot.draw_pc_sem_ins(sub_xyz, sub_labels, plot_colors)

if np.max(sub_rgb[:, 0:3]) > 20:  ## 0-255
    sub_rgb = sub_rgb / 255.0


sub_pc = np.concatenate([sub_xyz[:, 0:3], sub_rgb], axis=-1)

mask = (sub_labels == building_lable)
masked_pc = sub_pc[mask]

print("building_lable = ", building_lable)
print(len(masked_pc))

if len(masked_pc) > 0:
    Plot.draw_pc(masked_pc)
    #sem_labels = Plot.seg_pc(DATASET, masked_pc, eps = eps, min_points = 100, sep_vis = False, ori_color=False, do_recon = False)


