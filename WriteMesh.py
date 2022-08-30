import open3d as o3d
import numpy as np
import utm as ut

#convert UTM 2 Lon-Lat

UTM_OFFSET = [627285, 4841948, 0]
""" 
mesh = o3d.io.read_triangle_mesh("./output/p_mesh_1.ply")

sub_xyz = [ [*ut.to_latlon(vertex[0],vertex[1],17,'T'),vertex[2]] for vertex in (np.asarray(mesh.vertices) + UTM_OFFSET)]
print(np.min(np.asarray(sub_xyz).T[2]))

print(np.asarray(sub_xyz))

mesh.vertices = o3d.utility.Vector3dVector(np.asarray(sub_xyz))

if o3d.io.write_triangle_mesh("./output/p_mesh_1_lonlat.obj", mesh):
    print("success to transfer coordinates")
else:
    print("fail to transfer coordinates")


mesh = o3d.io.read_triangle_mesh("./output/p_mesh_1.obj")
print(np.asarray(mesh.vertices))
 """

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("remove points %s from total %s " % (len(outlier_cloud.points), len(cloud.points)))
    print("remove rate = %s %%" % (len(outlier_cloud.points)/len(cloud.points) * 100))

    print("Showing outliers (red) and inliers (gray): ")
    #outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    """     
    print(np.asarray(inlier_cloud.points))
    lon_lat_xyz = [ [*ut.to_latlon(vertex[0],vertex[1],17,'T'),vertex[2]] for vertex in (np.asarray(inlier_cloud.points) + UTM_OFFSET)]
    print(np.asarray(lon_lat_xyz))
    inlier_cloud.points = o3d.utility.Vector3dVector(lon_lat_xyz) """

    inlier_cloud.estimate_normals()
    poisson_mesh, densities= o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=15)


    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    bbox = inlier_cloud.get_axis_aligned_bounding_box()
    center = inlier_cloud.get_center()
    print(ut.to_latlon(center[0],center[1],17,'T'),center[2])
    p_mesh_crop = poisson_mesh.crop(bbox)

    o3d.visualization.draw_geometries([p_mesh_crop])
    """ o3d.io.write_triangle_mesh(save_ply_path, p_mesh_crop)
    o3d.io.write_triangle_mesh(save_obj_path, p_mesh_crop)"""
    
    o3d.io.write_triangle_mesh(save_gltf_path, p_mesh_crop) 


name = "TORONTO3D"

path = "./output/{}_pointcloud_1.ply".format(name)
save_ply_path = "./output/{}_mesh.ply".format(name)
save_obj_path = "./output/{}_mesh.obj".format(name)
save_gltf_path = "./output/{}_mesh.gltf".format(name)

pcd = o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pcd])
print("Statistical oulier removal")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=3.0)
display_inlier_outlier(pcd, ind)



