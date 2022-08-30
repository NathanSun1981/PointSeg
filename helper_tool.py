from cProfile import label
from turtle import color
import open3d as o3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd
import matplotlib.pyplot as plt
import utm as ut
import json


UTM_OFFSET = [627285, 4841948, 0]

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0): 
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb, label, name): 
        pc = o3d.geometry.PointCloud()
        print("setting points in o3d....")
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        print("setting colors in o3d....")
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        
        print("start to render ply...")
        o3d.visualization.draw_geometries([pc])
        return 0


    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None, label=None): 
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        print(sem_ins_labels)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]

            #print(valid_ind)
            if semins <= -1 or (label is not None and semins != label): 
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

    
            print(f"{tp} semins %s color" % semins)
            
            Y_colors[valid_ind] = tp

            ### bbox
            '''
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])
            '''
        print(Y_colors.shape)

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        print(Y_semins.shape)
        Plot.draw_pc(Y_semins)
        return Y_semins


    @staticmethod
    def seg_pc(name, pc_xyzrgb, eps = 1, min_points = 10, ori_color = False, sep_vis = False, do_recon = False): 

        pc = o3d.geometry.PointCloud()
        print("setting points in o3d....")
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] > 3:
            print("setting colors in o3d....")
            if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
                pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
            else:
                pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        print("start to seg ply...")

         #[INITIATION] 3D Shape Detection with RANSAC
        """ plane_model, inliers = pc.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = pc.select_by_index(inliers)
        outlier_cloud = pc.select_by_index(inliers, invert=True) """

        #print("Statistical oulier removal")
        #cl, ind = pc.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0) 
        #cl, ind = pc.remove_radius_outlier(nb_points=16, radius=0.05) 

        with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pc.cluster_dbscan(eps=eps, min_points = min_points, print_progress=True))

        sem_labels = np.unique(labels)
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        #Plot.display_inlier_outlier(pc, ind, ori_color)  
        pc_list = []

        if not sep_vis:
            if not ori_color:
                colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                colors[labels < 0] = 0
                pc.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([pc])
        else:
            for label in sem_labels:
                print("choose points with label %s" % label)
                if label > 0:
                    mask = (labels == label)
                    sem_pc = pc_xyzrgb[mask]
                    pc_list.append(sem_pc)
                    #print(pc_xyzrgb[mask].shape)
                    #Plot.draw_pc(sem_pc, label, name)
                    print("label = ", label)     
                    if do_recon:
                        Plot.reconstruction(sem_pc, label, name)                   
                                            
        return sem_labels
    
 
    @staticmethod
    def reconstruction(point_cloud, label, name): 

        save_ply_path = "./output/{}/ply/{}_mesh_{}.ply".format(name, name, str(label))
        save_obj_path = "./output/{}/obj/{}_mesh_{}.obj".format(name, name, str(label))
        save_gltf_path = "./output/{}/gltf/{}_mesh_{}.gltf".format(name, name, str(label))

        json_path = "./output/{}/obj/customTilesetOptions.json".format(name)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])

        #print(np.asarray(pcd.points)) 

        if np.max(point_cloud[:, 3:6]) > 20:  ## 0-255
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.)
        else:
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])

        print("start to write ply...")
        #o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud("./output/{}_pointcloud_{}.ply".format(name, str(label)), pcd)

        #estimate the reconstruction
        #PCD data will be export to reconstruction pipeline.
        print("Statistical oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=3.0)

        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)

        print("remove points %s from total %s " % (len(outlier_cloud.points), len(cl.points)))
        print("remove rate = %s %%" % (len(outlier_cloud.points)/len(cl.points) * 100))

        #bbox = inlier_cloud.get_axis_aligned_bounding_box()
        center = inlier_cloud.get_center()
        center_UTM = center + UTM_OFFSET
        center_lonlat = ut.to_latlon(center_UTM[0],center_UTM[1],17,'T')
        print(center_lonlat,center_UTM[2])
        #print(center)
        #print(np.asarray(inlier_cloud.points))

        rad = list(map(np.deg2rad,center_lonlat))

        the_revised_dict = Plot.get_json_data(json_path, rad[1], rad[0],  center_UTM[2])
        Plot.write_json_data(the_revised_dict, json_path)


        inlier_cloud.points = o3d.utility.Vector3dVector(inlier_cloud.points - center)
        bbox = inlier_cloud.get_axis_aligned_bounding_box()
        center = inlier_cloud.get_center()
        #print(center)
        #print(np.asarray(inlier_cloud.points))

        inlier_cloud.estimate_normals()
        poisson_mesh, densities= o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=9)

        print('remove low density vertices')
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
        
        #insert to database, with lables
        p_mesh_crop = poisson_mesh.crop(bbox)

        #print(np.asarray(p_mesh_crop.vertices))


        #o3d.visualization.draw_geometries([p_mesh_crop])
        o3d.io.write_triangle_mesh(save_ply_path, p_mesh_crop)
        o3d.io.write_triangle_mesh(save_obj_path, p_mesh_crop)
        o3d.io.write_triangle_mesh(save_gltf_path, p_mesh_crop) 

        cmd = "obj23dtiles -i {} --tileset -p {}".format(save_obj_path, json_path)
        os.system(cmd)

        #write obj to database

    @staticmethod
    def get_json_data(json_path, lon, lat, height):
        with open(json_path,'rb') as f:      
            params = json.load(f)           
            params['longitude'] = lon
            params['latitude'] = lat
            params['transHeight'] = height       
            dict = params           
        f.close()     
        return dict

    @staticmethod
    def write_json_data(dict, json_path):
        with open(json_path,'w') as r:
            json.dump(dict,r)           
        r.close()

