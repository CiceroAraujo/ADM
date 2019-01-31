import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
import time
import pyximport; pyximport.install()
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import scipy.sparse as sp

class Restriction:
    name_primal_tag = 'PRIMAL_ID_'
    name_fine_to_primal_tag = ['FINE_TO_PRIMAL', '_CLASSIC']
    name_wirebasket_id_tag = 'WIREBASKET_ID_LV'

    @staticmethod
    def get_or_nv1(mb, op, map_wirebasket, wirebasket_numbers):
        name_primal_tag_level = Restriction.name_primal_tag + str(1)
        name_fine_to_primal_tag_level = Restriction.name_fine_to_primal_tag[0] + str(1) + Restriction.name_fine_to_primal_tag[1]
        primal_tag = mb.tag_get_handle(name_primal_tag_level)
        fine_to_primal_tag = mb.tag_get_handle(name_fine_to_primal_tag_level)
        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]
        vertex_elems = np.array([item[0] for item in map_wirebasket.items() if item[1] >= ni+nf+ne])

        for elem in vertex_elems:
            gid = map_wirebasket[elem]
            line_op = op[gid]
            indice = sp.find(line_op)
            print(indice)
            import pdb; pdb.set_trace()


        OR = 0
        return OR


        # meshsets = self.mb.get_entities_by_type_and_tag(
        #     mb.get_root_set(), types.MBENTITYSET, np.array([primal_tag]),
        #     np.array([None]))
