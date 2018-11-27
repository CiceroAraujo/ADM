import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import pyximport; pyximport.install()
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import math
import os
import shutil
import random
import sys
import configparser




name_inputfile = '9x36x36-volume.vtk'
#
# mesh_config_file = 'mesh_configs.cfg'
# config = configparser.ConfigParser()
# config.read(mesh_config_file)
# total_dimension = config['total-dimension']
# Lx = long(total_dimension['Lx'])
# Ly = long(total_dimension['Ly'])
# Lz = long(total_dimension['Lz'])
# import pdb; pdb.set_trace()


Lx = 36
Ly = 36
Lz = 9

l1 = 3
l2 = 9

comm = Epetra.PyComm()
mb = core.Core()
mb.load_file(name_inputfile)
mtu = topo_util.MeshTopoUtil(mb)
root_set = mb.get_root_set()
all_volumes = mb.get_entities_by_dimension(root_set, 3)
nf = len(all_volumes)
nc1 = Lx*Ly*Lz/(l1**3)
nc2 = Lx*Ly*Lz/(l2**3)


intern_elems_meshset = mb.create_meshset()
face_elems_meshset = mb.create_meshset()
edge_elems_meshset = mb.create_meshset()
vertex_elems_meshset = mb.create_meshset()
D1_tag = mb.tag_get_handle('d1')

for v in all_volumes:
    value = mb.tag_get_data(D1_tag, v, flat=True)[0]
    if value == 0:
        mb.add_entities(intern_elems_meshset, [v])
    elif value == 1:
        mb.add_entities(face_elems_meshset, [v])
    elif value == 2:
        mb.add_entities(edge_elems_meshset, [v])
    elif value == 3:
        mb.add_entities(vertex_elems_meshset, [v])
    else:
        print('Erro de tags')
        print(v)
        sys.exit(0)


intern_elems = mb.get_entities_by_handle(intern_elems_meshset)
face_elems = mb.get_entities_by_handle(face_elems_meshset)
edge_elems = mb.get_entities_by_handle(edge_elems_meshset)
vertex_elems = mb.get_entities_by_handle(vertex_elems_meshset)

elems_wirebasket = intern_elems + face_elems + edge_elems + vertex_elems

#criando tags



#carregando as tags
perm_tag = mb.tag_get_handle("PERM")
press_tag = mb.tag_get_handle("P")
q_tag = mb.tag_get_handle("Q")
wells_tag = mb.tag_get_handle("WELLS")
wells_d_tag = mb.tag_get_handle("WELLS_D")
wells_n_tag = mb.tag_get_handle("WELLS_N")
all_faces_boundary_tag = mb.tag_get_handle("FACES_BOUNDARY")

wells_n = mb.tag_get_data(wells_n_tag)
wells_d = mb.tag_get_data(wells_d_tag)
press = mb.tag_get_data(press_tag, wells_d)
vazao = mb.tag_get_data(q_tag, wells_n)

map_global_wirebasket = dict(zip(elems_wirebasket, range(nf)))

all_faces = mb.get_entities_by_dimension(root_set, 2)

def get_local_matrix(self, face, map_local, **options):
        """
        obtem a matriz local e os elementos correspondentes
        se flag == 1 retorna o fluxo multiescala entre dois elementos separados pela face
        """


        elems = self.mb.get_adjacencies(face, 3)

        k1 = self.mb.tag_get_data(perm_tag, elems[0]).reshape([3, 3])
        k2 = self.mb.tag_get_data(perm_tag, elems[1]).reshape([3, 3])
        centroid1 = mtu.get_average_position([elems[0]])
        centroid2 = mtu.get_average_position([elems[1]])
        direction = centroid2 - centroid1
        uni = self.unitary(direction)
        k1 = np.dot(np.dot(k1,uni),uni)
        k2 = np.dot(np.dot(k2,uni),uni)
        keq = self.kequiv(k1, k2)*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))

        return -keq, elems, adjs

def unitary(v):
    a = v/float(np.linalg.norm(v))
    return a*a

def set_global_problem_AMS_gr_numpy_to_OP(map_global):
    """
    transmissibilidade da malha fina
    obs: com funcao para obter dados dos elementos
    """
    #0
    dict_wells_n = dict(zip(wells_n, vazao))
    dict_wells_d = dict(zip(wells_d, press))

    trans_fine = np.zeros((nf, nf), dtype='float64')
    b = np.zeros(nf, dtype='float64')
    s = b.copy()
    for volume in set(all_volumes):
        #1
        temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
        trans_fine[map_global[volume], temp_glob_adj] = temp_k
        b[map_global[volume]] += source_grav
        s[map_global[volume]] = source_grav
        if volume in self.wells_n:
            #2
            if volume in self.wells_inj:
                #3
                b[map_global[volume]] += dict_wells_n[volume]
            #2
            else:
                #3
                b[map_global[volume]] += -dict_wells_n[volume]
    #0
    # for volume in self.wells_d:
    #     #1
    #     # temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
    #     # self.mb.tag_set_data(self.flux_coarse_tag, volume, source_grav)
    #     trans_fine[map_global[volume], map_global[volume]] = 1.0
    #     b[map_global[volume]] = dict_wells_d[volume]

    return trans_fine, b, s

def mount_lines(face):
