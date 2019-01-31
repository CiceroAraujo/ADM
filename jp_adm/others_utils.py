import numpy as np
# from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
# import time
import pyximport; pyximport.install()
# from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
# import math
# import os
# import shutil
# import random
import sys
# import configparser
import io
import yaml
from pymoab_utils import UtilsPymoab as utpy

class OtherUtils:
    name_keq_tag = 'K_EQ'
    name_s_grav = 'S_GRAV'
    name_dirichlet_tag = 'P'
    name_neumann_tag = 'Q'
    list_names_tags = [name_keq_tag, name_s_grav, name_dirichlet_tag, name_neumann_tag]

    def __init__(self, mb=None, comm=None):
        if mb == None:
            raise RuntimeError('Defina o mb core do pymoab')
        elif comm == None:
            raise RuntimeError('Defina o comm do trilinos')

        self.list_tags = [mb.tag_get_handle(name) for name in OtherUtils.list_names_tags]
        self.mb = mb
        self.comm = comm
        self.mtu = topo_util.MeshTopoUtil(mb)
        self.gravity = True
        # return super().__new__(cls)

    def mount_local_problem(self, map_local, faces_in=None):
        """
        retorna os indices: linha, coluna, valor, sz
        sz = tamanho da matriz
        input:
            map_local: dicionario onde keys = volumes e values = id local
            faces_in: faces internas do conjunto de volumes
        output:
            inds:
        """
        elements = rng.Range(list(map_local.keys()))
        n = len(elements)

        if faces_in == None:
            # faces = utpy.get_all_faces(OtherUtils.mb, rng.Range(elements))
            # bound_faces = utpy.get_boundary_of_volumes(OtherUtils.mb, elements)
            # faces = rng.subtract(faces, bound_faces)
            faces = rng.subtract(utpy.get_faces(self.mb, rng.Range(elements)), utpy.get_boundary_of_volumes(self.mb, elements))
        else:
            faces = faces_in

        keqs = mb.tag_get_data(self.list_tags[0], faces, flat=True)
        elems = [self.mb.get_adjacencies(face) for face in faces]
        s_gravs = self.mb.tag_get_data(self.list_tags[1], faces, flat=True)
        dict_keq = dict(zip(faces, keqs))
        dict_elems = dict(zip(faces, elems))
        dict_s_grav = dict(zip(faces, s_gravs))

        linesM = np.array([])
        colsM = np.array([])
        valuesM = np.array([])
        linesM2 = np.array([])
        valuesM2 = np.array([])
        szM = [n, n]

        b = np.zeros(n)
        s = np.zeros(n)

        for face in faces:
            elems = dict_elems[face]
            keq = dict_keq[face]
            s_grav = dict_s_grav[face]

            linesM = np.append(linesM, [map_local[elems[0]], map_local[elems[1]]])
            colsM = np.append(colsM, [map_local[elems[1]], map_local[elems[0]]])
            valuesM = np.append(valuesM, [-keq, -keq])

            ind0 = np.where(linesM2 == map_local[elems[0]])
            if len(ind0[0]) == 0:
                linesM2 = np.append(linesM2, map_local[elems[0]])
                valuesM2 = np.append(valuesM2, [keq])
            else:
                valuesM2[ind0[0]] += keq

            ind1 = np.where(linesM2 == map_local[elems[1]])
            if len(ind1[0]) == 0:
                linesM2 = np.append(linesM2, map_local[elems[1]])
                valuesM2 = np.append(valuesM2, [keq])
            else:
                valuesM2[ind1[0]] += keq

            s[map_local[elems[0]]] += s_grav
            s[map_local[elems[1]]] -= s_grav

        linesM = np.append(linesM, linesM2)
        colsM = np.append(colsM, linesM2)
        valuesM = np.append(valuesM, valuesM2)

        linesM = linesM.astype(np.int32)
        colsM = colsM.astype(np.int32)

        inds = np.array([linesM, colsM, valuesM, szM, False, False])

        if self.gravity == True:
            return inds, s
        else:
            return inds, b

    def get_flow_on_boundary(self, elements, p_tag, bound_faces=None):
        """
        input:
            elements: volumes ou meshset de volumes
            p_tag = tag da pressao
            bound_faces = faces no contorno do volume
        output:
            dict com keys = elements e values = fluxo na face do contorno
        """
        dict_flux = {}
        elements = utpy.get_elements(elements)
        if bound_faces  == None:
            faces = utpy.get_boundary_of_volumes(self.mb, elements)
        else:
            faces = bound_faces

        keqs = self.mb.tag_get_data(OtherUtils.list_tags[0], faces, flat=True)
        elems = [self.mb.get_adjacencies(face) for face in faces]
        s_gravs = self.mb.tag_get_data(OtherUtils.list_tags[1], faces, flat=True)
        dict_keq = dict(zip(faces, keqs))
        dict_elems = dict(zip(faces, elems))
        dict_s_grav = dict(zip(faces, s_gravs))

        for face in faces:
            elems = dict_elems[face]
            keq = dict_keq[face]
            s_grav = dict_s_grav[face]

            p = self.mb.tag_get_data(p_tag, elems, flat=True)
            flux = (p[1] - p[0])
            if OtherUtils.gravity == True:
                flux += s_grav

            if elems[0] in elements:
                dict_flux[elems[0]] = flux
            else:
                dict_flux[elems[1]] = -flux

        return dict_flux

    @staticmethod
    def set_boundary_dirichlet(map_local, map_values, b, inds):
        inds2 = inds.copy()
        for v in boundary_elems:
            gid = map_local[v]
            indices = np.where(inds2[0] == gid)[0]
            inds2[0] = np.delete(inds2[0], indices)
            inds2[1] = np.delete(inds2[1], indices)
            inds2[2] = np.delete(inds2[2], indices)

            inds2[0] = np.append(inds2[0], np.array([gid]))
            inds2[1] = np.append(inds2[1], np.array([gid]))
            inds2[2] = np.append(inds2[2], np.array([1.0]))
            b[gid] = map_values[v]

        inds2[0] = inds2[0].astype(np.int32)
        inds2[1] = inds2[1].astype(np.int32)

        return inds2, b

    @staticmethod
    def set_boundary_neumann(map_local, map_values, b):
        for v in boundary_elems:
            gid = map_local[v]
            b[gid] += map_values[v]

        return b

    @staticmethod
    def get_slice_by_inds(inds, slice_row, slice_col):
        """
        retorna um slice a partir dos inds
        input:
            inds
            slice_row
            slice_col
        output:
            inds2:
        """

        lines2 = np.array([])
        cols2 = np.array([])
        values2 = np.array([], dtype=np.float64)

        map_l = dict(zip(slice_rows, range(len(slice_rows))))
        map_c = dict(zip(slice_cols, range(len(slice_cols))))
        sz = [len(slice_rows), len(slice_cols)]

        for i in slice_rows:
            assert i in inds[0]
            indices = np.where(inds[0] == i)[0]
            cols = np.array([inds[1][j] for j in indices if inds[1][j] in slice_cols])
            vals = np.array([inds[2][j] for j in indices if inds[1][j] in slice_cols])
            lines = np.repeat(i, len(cols))

            lines2 = np.append(lines2, lines)
            cols2 = np.append(cols2, cols)
            values2 = np.append(values2, vals)

        lines2 = lines2.astype(np.int32)
        cols2 = cols2.astype(np.int32)
        local_inds_l = np.array([map_l[j] for j in lines2]).astype(np.int32)
        local_inds_c = np.array([map_c[j] for j in cols2]).astype(np.int32)

        inds2 = np.array([lines2, cols2, values2, sz, local_inds_l, local_inds_c])
        return inds2

    @staticmethod
    def get_tc_tpfa(inds_tc_wirebasket, wirebasket_numbers):
        """
        obtem a 'tpfa-lização' da matriz mpfa
        input:
            wirebasket_numbers = [ni, nf, ne, nv]
            inds_tc_wirebasket: mpfa wirebasket transmissibility  of coarse level
        output:
            inds_tc_tpfa
        """
        import pdb; pdb.set_trace()
        nni = wirebasket_numbers[0]
        nnf = wirebasket_numbers[1] + nni
        nne = wirebasket_numbers[2] + nnf
        nnv = wirebasket_numbers[3] + nne

        inds_tc_tpfa = inds_tc_wirebasket.copy()

        inds_tc_tpfa = OtherUtils.indices_maiores(inds_tc_tpfa, 0, nni, nnf)
        inds_tc_tpfa = OtherUtils.indices_maiores(inds_tc_tpfa, nni, nnf, nne)
        inds_tc_tpfa = OtherUtils.indices_menores(inds_tc_tpfa, nnf, nne, nni)
        inds_tc_tpfa = OtherUtils.indices_menores(inds_tc_tpfa, nne, nnv, nnf)

        return inds_tc_tpfa

    @staticmethod
    def get_tmod_by_inds(inds_wirebasket, wirebasket_numbers):
        """
        obtem a transmissibilidade wirebasket modificada
        """
        # ordem: vertex, edge, face, intern
        inds = inds_wirebasket
        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]

        lines2 = np.array([], dtype=np.int32)
        cols2 = lines2.copy()
        values2 = np.array([], dtype='float64')

        lines = set(inds[0])
        sz = inds[3][:]

        verif1 = ni
        verif2 = ni+nf
        rg1 = np.arange(ni, ni+nf)

        for i in lines:
            indice = np.where(inds[0] == i)[0]
            if i < ni:
                lines2 = np.hstack((lines2, inds[0][indice]))
                cols2 = np.hstack((cols2, inds[1][indice]))
                values2 = np.hstack((values2, inds[2][indice]))
                continue
            elif i >= ni+nf+ne:
                continue
            elif i in rg1:
                verif = verif1
            else:
                verif = verif2

            lines_0 = inds[0][indice]
            cols_0 = inds[1][indice]
            vals_0 = inds[2][indice]

            inds_minors = np.where(cols_0 < verif)[0]
            vals_minors = vals_0[inds_minors]

            vals_0[np.where(cols_0 == i)[0]] += sum(vals_minors)
            inds_sup = np.where(cols_0 >= verif)[0]
            lines_0 = lines_0[inds_sup]
            cols_0 = cols_0[inds_sup]
            vals_0 = vals_0[inds_sup]


            lines2 = np.hstack((lines2, lines_0))
            cols2 = np.hstack((cols2, cols_0))
            values2 = np.hstack((values2, vals_0))

        lines2 = lines2.astype(np.int32)
        cols2 = cols2.astype(np.int32)

        inds2 = np.array([lines2, cols2, values2, sz])

        return inds2

    @staticmethod
    def indices_maiores(inds_tc_tpfa, ind1, ind2, verif):

        for i in range(ind1, ind2):
            indices = np.where(inds_tc_tpfa[0] == i)[0]
            line = inds_tc_tpfa[0][indices]
            cols = inds_tc_tpfa[1][indices]
            values = inds_tc_tpfa[2][indices]

            inds_tc_tpfa[0] = np.delete(inds_tc_tpfa[0], indices)
            inds_tc_tpfa[1] = np.delete(inds_tc_tpfa[1], indices)
            inds_tc_tpfa[2] = np.delete(inds_tc_tpfa[2], indices)

            indices = np.where(cols >= verif)[0]

            # soma = values[indices].sum()
            # values[cols == i] += soma
            values[cols == i] += values[indices].sum()
            line = np.delete(line, indices)
            cols = np.delete(cols, indices)
            values = np.delete(values, indices)

            inds_tc_tpfa[0] = np.append(inds_tc_tpfa[0], line)
            inds_tc_tpfa[1] = np.append(inds_tc_tpfa[1], cols)
            inds_tc_tpfa[2] = np.append(inds_tc_tpfa[2], values)

        return inds_tc_tpfa

    @staticmethod
    def indices_menores(inds_tc_tpfa, ind1, ind2, verif):
        for i in range(ind1, ind2):
            indices = np.where(inds_tc_tpfa[0] == i)[0]
            line = inds_tc_tpfa[0][indices]
            cols = inds_tc_tpfa[1][indices]
            values = inds_tc_tpfa[2][indices]

            inds_tc_tpfa[0] = np.delete(inds_tc_tpfa[0], indices)
            inds_tc_tpfa[1] = np.delete(inds_tc_tpfa[1], indices)
            inds_tc_tpfa[2] = np.delete(inds_tc_tpfa[2], indices)

            indices = np.where(cols < verif)[0]

            # soma = values[indices].sum()
            # values[cols == i] += soma
            values[cols == i] += values[indices].sum()
            line = np.delete(line, indices)
            cols = np.delete(cols, indices)
            values = np.delete(values, indices)

            inds_tc_tpfa[0] = np.append(inds_tc_tpfa[0], line)
            inds_tc_tpfa[1] = np.append(inds_tc_tpfa[1], cols)
            inds_tc_tpfa[2] = np.append(inds_tc_tpfa[2], values)

        return inds_tc_tpfa
