import open3d as o3d
import numpy as np
import utm as ut
import sys, getopt

#convert UTM 2 Lon-Lat


def main(argv):
    UTM_OFFSET = [627285, 4841948, 0]
    inputfile = ''
    outputfile = ''
    utm = [0,0,0]
    need_labels = False
    need_color = False
    try:
        opts, args = getopt.getopt(argv,"hi:o:lu:c",["ifile=","ofile=","utm="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-l", "--lables"):
            need_labels = True
        elif opt in ("-c", "--colors"):
            need_color = True
        elif opt in ("-u", "--utm"):          
            utm = arg.split(',')
            print(list(map(int, utm)))
        

    if inputfile == '' or outputfile == '':
        print ('input output must be assigned')
        sys.exit()

    if inputfile[-3:] != 'ply':
        print ('input must be ply!')
        sys.exit()
    
    if outputfile[-3:] != 'txt':
        print ('input must be txt!')
        sys.exit()

    data = o3d.t.io.read_point_cloud(inputfile).point
    points = data["positions"].numpy()
    points = np.float32(points + list(map(int, utm)))
    feat = data["colors"].numpy().astype(np.float32)
    if need_labels:
        labels = data['scalar_Label'].numpy().astype(np.int32)
        if need_color:
            txt_data = np.concatenate((np.asarray(points), np.asarray(feat), np.asarray(labels)), axis=1)
        else:
            txt_data = np.concatenate((np.asarray(points), np.asarray(labels)), axis=1)
    else:
        if need_color:
            txt_data = np.concatenate((np.asarray(points), np.asarray(feat)), axis=1)
        else:
            txt_data = np.asarray(points)
    """ points = np.float32(pcd.point["positions"].numpy())
    colors = np.int32(pcd.point["colors"].numpy())
    labels = np.int32(pcd.point["class"].numpy()) """
    #print(np.asarray(points))

    
    #print(txt_data)

    with open(outputfile, 'w') as my_file:
        np.savetxt(my_file, txt_data, fmt='%.2f', delimiter=' ')

    print('Array exported to file')
    #pcd = o3d.t.io.write_point_cloud("./output/L002.xyzrgb", pcd)

if __name__ == "__main__":
   main(sys.argv[1:])