import psycopg2
import gpudb
import numpy as np
import json
import os

url = "http://206.12.92.18:10091"
username = "gtfs_admin"
password = "GTFS_Admin2022"

kdb = gpudb.GPUdb(host = url, username = username, password = password)

schema = "ki_home"
lidar_table = schema + ".3dtiles"

table_lidar_obj = gpudb.GPUdbTable(_type=None, name=lidar_table, db=kdb)

class TileImporter:
    @staticmethod
    def convert_to_polygon(name, file_path):
        
        if not os.path.exists(file_path):
            print("No such json file!")
            return
        f = open(file_path)
        data = json.load(f)
        radians = list(map(np.rad2deg, data['root']['boundingVolume']['region'][0:4]))
        d = {'west': radians[0], 'south': radians[1], 'east': radians[2], 'north': radians[3]}

        polygon_data = 'POLYGON(('
        polygon_data += f'{d["west"]} {d["north"]},'
        polygon_data += f'{d["east"]} {d["north"]},'
        polygon_data += f'{d["east"]} {d["south"]},'
        polygon_data += f'{d["west"]} {d["south"]},'
        polygon_data += f'{d["west"]} {d["north"]},'
        polygon_data = polygon_data[:-1] + '))'
        cur.execute(f'INSERT INTO public.tiles_3d (name, path, geo_polygon) VALUES (\'{name}\', \'{polygon_data}\');')
        conn.commit()
        conn.close()

