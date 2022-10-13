import os
import torch
import numpy as np
from pytorch3d.io import IO
from pytorch3d.io import load_obj, save_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")



def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()


trg_obj = os.path.join(os.getcwd(), "output/TORONTO3D_pointcloud_1.ply")

verts, faces = load_ply(trg_obj)
#verts, faces, aux = load_obj(trg_obj)
print(verts,faces)

# Load point cloud

point_cloud = Pointclouds(points=[verts], features=[rgb])
""" faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale """

# We construct a Meshes structure for the target mesh
#trg_mesh = Meshes(verts=[verts], faces=[faces])

#src_mesh = ico_sphere(4, device)
#plot_pointcloud(trg_mesh, "Target mesh")
#plot_pointcloud(src_mesh, "Source mesh")