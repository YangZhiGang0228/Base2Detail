from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from include import *

def main():
    mesh_folders = ['./data_meshes/example/']
    S = TrainMeshes(mesh_folders)

    pickle.dump(S, file = open("./data_PKL/train.pkl", "wb"))
if __name__ == '__main__':
    main()
