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




name_inputfile = '9x27x27.h5m'
#
# mesh_config_file = 'mesh_configs.cfg'
# config = configparser.ConfigParser()
# config.read(mesh_config_file)
# total_dimension = config['total-dimension']
# Lx = long(total_dimension['Lx'])
# Ly = long(total_dimension['Ly'])
# Lz = long(total_dimension['Lz'])
# import pdb; pdb.set_trace()



class AMS_prol:

    def __init__(self, inputfile):

        self.comm = Epetra.PyComm()
        self.mb = core.Core()
        self.mb.load_file(inputfile)
        self.mtu = topo_util.MeshTopoUtil(self.mb)
        self.root_set = self.mb.get_root_set()
        self.all_volumes = self.mb.get_entities_by_dimension(self.root_set, 3)
        self.all_faces = self.mb.get_entities_by_dimension(self.root_set, 2)
        self.nf = len(self.all_volumes)
        self.create_tags()
        self.create_elems_wirebasket()
        self.map_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))
        self.get_wells()
        self.primals1 = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.prilmal_ids1_tag]),
            np.array([None]))
        self.nc1 = len(self.primals1)
        self.primals2 = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.prilmal_ids2_tag]),
            np.array([None]))

        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat=True)
        minim = min(all_gids)
        all_gids = all_gids - minim
        self.mb.tag_set_data(self.global_id0_tag, self.all_volumes, all_gids)

        ids_nv1 = self.mb.tag_get_data(self.L1_tag, self.all_volumes, flat=True)
        ids_nv1 -= 1
        self.mb.tag_set_data(self.L1_tag, self.all_volumes, ids_nv1)

        Lx = 27
        Ly = 27
        Lz = 9

        self.tz = Lz
        self.gama = 10.0
        self.mi = 1.0

        l1 = 3
        l2 = 9

        nc1 = Lx*Ly*Lz/(l1**3)
        nc2 = Lx*Ly*Lz/(l2**3)

        self.run()

    def create_elems_wirebasket(self):

        intern_elems_meshset = self.mb.create_meshset()
        face_elems_meshset = self.mb.create_meshset()
        edge_elems_meshset = self.mb.create_meshset()
        vertex_elems_meshset = self.mb.create_meshset()

        for v in self.all_volumes:
            value = self.mb.tag_get_data(self.D1_tag, v, flat=True)[0]
            if value == 0:
                self.mb.add_entities(intern_elems_meshset, [v])
            elif value == 1:
                self.mb.add_entities(face_elems_meshset, [v])
            elif value == 2:
                self.mb.add_entities(edge_elems_meshset, [v])
            elif value == 3:
                self.mb.add_entities(vertex_elems_meshset, [v])
            else:
                print('Erro de tags')
                print(v)
                sys.exit(0)

        self.intern_elems = list(self.mb.get_entities_by_handle(intern_elems_meshset))
        self.face_elems = list(self.mb.get_entities_by_handle(face_elems_meshset))
        self.edge_elems = list(self.mb.get_entities_by_handle(edge_elems_meshset))
        self.vertex_elems = list(self.mb.get_entities_by_handle(vertex_elems_meshset))

        self.elems_wirebasket = self.intern_elems + self.face_elems + self.edge_elems + self.vertex_elems

    def create_tags(self):
        self.global_id0_tag = self.mb.tag_get_handle('GLOBAL_ID')
        self.D1_tag = self.mb.tag_get_handle('d1')
        self.D2_tag = self.mb.tag_get_handle('d2')
        self.L1_tag = self.mb.tag_get_handle('l1_ID')
        self.L2_tag = self.mb.tag_get_handle('l2_ID')
        self.L3_tag = self.mb.tag_get_handle('l3_ID')
        self.prilmal_ids1_tag = self.mb.tag_get_handle('PRIMAL_ID_1')
        self.prilmal_ids2_tag = self.mb.tag_get_handle('PRIMAL_ID_2')
        self.perm_tag = self.mb.tag_get_handle("PERM")
        self.press_tag = self.mb.tag_get_handle("P")
        self.q_tag = self.mb.tag_get_handle("Q")
        self.wells_tag = self.mb.tag_get_handle("WELLS")
        self.wells_d_tag = self.mb.tag_get_handle("WELLS_D")
        self.wells_n_tag = self.mb.tag_get_handle("WELLS_N")
        self.all_faces_boundary_tag = self.mb.tag_get_handle("FACES_BOUNDARY")
        self.area_tag = self.mb.tag_get_handle("AREA")
        self.fine_to_primal1_classic_tag = self.mb.tag_get_handle("FINE_TO_PRIMAL1_CLASSIC")
        self.fine_to_primal2_classic_tag = self.mb.tag_get_handle("FINE_TO_PRIMAL2_CLASSIC")
        self.pf_tag = self.mb.tag_get_handle("PF", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        # self.id_wells_tag = self.mb.tag_get_handle("I", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    def get_CrsMatrix_by_array(self, M, n_rows = None, n_cols = None):
        """
        retorna uma CrsMatrix a partir de um array numpy
        input:
            M: array numpy (matriz)
            n_rows: (opcional) numero de linhas da matriz A
            n_cols: (opcional) numero de colunas da matriz A
        output:
            A: CrsMatrix
        """

        if n_rows == None and n_cols == None:
            rows, cols = M.shape
        else:
            if n_rows == None or n_cols == None:
                print('determine n_rows e n_cols')
                sys.exit(0)
            else:
                rows = n_rows
                cols = n_cols

        row_map = Epetra.Map(rows, 0, self.comm)
        col_map = Epetra.Map(cols, 0, self.comm)
        A = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 7)

        rows = np.nonzero(M)[0].astype(np.int32)
        cols = np.nonzero(M)[1].astype(np.int32)

        if self.verif == True:
            print(rows)
            print(cols)
            print(M[rows, cols])
            import pdb; pdb.set_trace()

        A.InsertGlobalValues(rows, cols, M[rows, cols])

        return A

    def get_CrsMatrix_by_inds(self, inds):
        """
        retorna uma CrsMatrix a partir de inds
        input:
            inds: array numpy com informacoes da matriz
        output:
            A: CrsMatrix
        """

        rows = inds[3][0]
        cols = inds[3][1]

        row_map = Epetra.Map(rows, 0, self.comm)
        col_map = Epetra.Map(cols, 0, self.comm)
        A = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 7)

        A.InsertGlobalValues(inds[0], inds[1], inds[2])

        return A

    def get_inverse_tril(self, A, rows):
        """
        Obter a matriz inversa de A
        obs: A deve ser quadrada
        input:
            A: CrsMatrix
            rows: numero de linhas

        output:
            INV: CrsMatrix inversa de A
        """
        num_cols = A.NumMyCols()
        num_rows = A.NumMyRows()
        assert num_cols == num_rows
        map1 = Epetra.Map(rows, 0, self.comm)

        Inv = Epetra.CrsMatrix(Epetra.Copy, map1, 3)

        for i in range(rows):
            b = Epetra.Vector(map1)
            b[i] = 1.0

            x = self.solve_linear_problem(A, b, rows)
            lines = np.nonzero(x[:])[0].astype(np.int32)
            col = np.repeat(i, len(lines)).astype(np.int32)
            Inv.InsertGlobalValues(lines, col, x[lines])

        return Inv

    def get_kequiv_by_face_quad(self, face):
        """
        retorna os valores de k equivalente para colocar na matriz
        a partir da face

        input:
            face: face do elemento
        output:
            kequiv: k equivalente
            elems: elementos vizinhos pela face
            s: termo fonte da gravidade
        """

        elems = self.mb.get_adjacencies(face, 3)
        k1 = self.mb.tag_get_data(self.perm_tag, elems[0]).reshape([3, 3])
        k2 = self.mb.tag_get_data(self.perm_tag, elems[1]).reshape([3, 3])
        centroid1 = self.mtu.get_average_position([elems[0]])
        centroid2 = self.mtu.get_average_position([elems[1]])
        direction = centroid2 - centroid1
        uni = self.unitary(direction)
        k1 = np.dot(np.dot(k1,uni), uni)
        k2 = np.dot(np.dot(k2,uni), uni)
        area = self.mb.tag_get_data(self.area_tag, face, flat=True)[0]
        keq = self.kequiv(k1, k2)*area/(self.mi*np.linalg.norm(direction))
        z1 = self.tz - centroid1[2]
        z2 = self.tz - centroid2[2]
        s_gr = self.gama*keq*(z1-z2)

        return keq, s_gr, elems

    def get_negative_matrix(self, matrix, n):
        std_map = Epetra.Map(n, 0, self.comm)
        if matrix.Filled() == False:
            matrix.FillComplete()
        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        EpetraExt.Add(matrix, False, -1.0, A, 1.0)

        return A

    def get_OP(self):
        self.verif = False
        lim = 1e-7

        # map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))
        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)

        self.inds_OP = np.array([np.array([]), np.array([]), np.array([],dtype = np.float64), [self.nf, self.nc1]])

        idsi = ni
        idsf = ni+nf
        idse = idsf+ne
        idsv = idse+nv

        std_map = Epetra.Map(self.nf, 0, self.comm)
        self.OP = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

        # self.put_matrix_into_OP(np.identity(nv, dtype='float64'), nv, ni+nf+ne, ni+nf+ne+nv)

        ind1 = idse
        ind2 = idsv

        ident = np.identity(nv, dtype=np.float64)
        lines = np.nonzero(ident)[0].astype(np.int32)
        cols = np.nonzero(ident)[1].astype(np.int32)
        values = ident[lines, cols]
        sz = [nv, nv]
        inds_0 = np.array([lines, cols, values, sz])
        self.put_indices_into_OP(inds_0, ind1, ind2)


        ###
        #elementos de aresta (edge)
        ind1 = idsf
        ind2 = idse
        # import pdb; pdb.set_trace()
        M = self.get_CrsMatrix_by_array(self.trans_mod[idsf:idse, idsf:idse])
        M = self.get_inverse_tril(M, ne)
        M = self.get_negative_matrix(M, ne)
        M2 = self.get_CrsMatrix_by_array(self.trans_mod[idsf:idse, idse:idsv], n_rows = ne, n_cols = ne)
        M = self.pymultimat(M, M2, ne)
        M2, indsM2 = self.modificar_matriz(M, ne, nv, ne, return_inds = True)
        # self.put_CrsMatrix_into_OP(M2, ind1, ind2)
        self.put_indices_into_OP(indsM2, ind1, ind2)
        # self.test_OP_tril(ind1 = idsf, ind2 = idse)



        #elementos de face
        if nf > ne:
            nvols = nf
        else:
            nvols = ne
        ind1 = idsi
        ind2 = idsf
        M2 = self.get_CrsMatrix_by_array(self.trans_mod[idsi:idsf, idsi:idsf])
        M2 = self.get_inverse_tril(M2, nf)
        M2 = self.get_negative_matrix(M2, nf)
        M2 = self.modificar_matriz(M2, nvols, nvols, nf)
        M3 = self.get_CrsMatrix_by_array(self.trans_mod[idsi:idsf, idsf:idse], n_rows = nvols, n_cols = nvols)
        M = self.modificar_matriz(M, nvols, nvols, ne)
        M = self.pymultimat(self.pymultimat(M2, M3, nvols), M, nvols)
        M2, indsM2 = self.modificar_matriz(M, nf, nv, nf, return_inds = True)
        # self.put_CrsMatrix_into_OP(M2, ind1, ind2)
        self.put_indices_into_OP(indsM2, ind1, ind2)

        # self.test_OP_tril(ind1 = idsi, ind2 = idsf)


        #elementos internos
        if ni > nf:
            nvols = ni
        else:
            nvols = nf

        ind1 = 0
        ind2 = idsi
        M2 = self.get_CrsMatrix_by_array(self.trans_mod[0:idsi, 0:idsi])   #A
        M2 = self.get_inverse_tril(M2, ni)                                 #B
        M2 = self.get_negative_matrix(M2, ni)
        M2 = self.modificar_matriz(M2, nvols, nvols, ni)
        M3 = self.get_CrsMatrix_by_array(self.trans_mod[0:idsi, idsi:idsf], n_rows = nvols, n_cols = nvols) #D
        M = self.modificar_matriz(M, nvols, nvols, nf)                                                       #E
        M = self.pymultimat(self.pymultimat(M2, M3, nvols), M, nvols)                                       #F
        M2, indsM2 = self.modificar_matriz(M, ni, nv, ni, return_inds = True)                         #G

        #OP[0:idsi] = np.dot(np.dot(C, self.trans_mod[0:idsi, idsi:idsf]), OP[idsi:idsf])

        # self.put_CrsMatrix_into_OP(M2, ind1, ind2)
        self.put_indices_into_OP(indsM2, ind1, ind2)
        # self.test_OP_tril(ind1 = 0, ind2 = idsi)

        self.OP = self.pymultimat(self.G, self.OP, self.nf)
        op, self.inds_OP = self.modificar_matriz(self.OP, self.nf, self.nc1, self.nf, return_inds = True)
        np.save('inds_op', self.inds_OP)

    def get_OR1(self):

        linesf = np.array([])
        colsf = np.array([])
        valuesf = np.array([], dtype=np.float64)
        n_cols=0
        n_lines = self.nf

        my_primals1 = set()

        for v in self.all_volumes:
            nv = self.mb.tag_get_data(self.L3_tag, v, flat=True)[0]
            if nv == 1 or nv == 2:
                gid_nv1 = self.mb.tag_get_data(self.L1_tag, v, flat = True)[0]
                if gid_nv1 in my_primals1:
                    continue
                or_tag = self.mb.tag_get_handle(
                    "OR_{0}".format(gid_nv1), 1, types.MB_TYPE_INTEGER, True,
                    types.MB_TAG_SPARSE, default_value=0)
                my_primals1.add(gid_nv1)
                elems_in_primal = self.mb.get_entities_by_type_and_tag(
                    self.root_set, types.MBHEX, self.L1_tag,
                    np.array([gid_nv1]))

                gids_elems = self.mb.tag_get_data(self.global_id0_tag, elems_in_primal, flat=True)

                linesf = np.append(linesf, gids_elems)
                colsf = np.append(colsf, np.repeat(gid_nv1, len(gids_elems)))
                valuesf = np.append(valuesf, np.ones(len(gids_elems)))
                n_cols+=1
                self.mb.tag_set_data(or_tag, elems_in_primal, np.ones(len(elems_in_primal), dtype=np.int))
            else:
                gid_nv1 = self.mb.tag_get_data(self.L1_tag, v, flat = True)
                gid_elem = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)
                or_tag = self.mb.tag_get_handle(
                    "OR_{0}".format(gid_nv1), 1, types.MB_TYPE_INTEGER, True,
                    types.MB_TAG_SPARSE, default_value=0)

                linesf = np.append(linesf, gid_elem)
                colsf = np.append(colsf, gid_nv1)
                valuesf = np.append(valuesf, np.array([1.0]))
                n_cols+=1
                self.mb.tag_set_data(or_tag, v, 1)


        linesf = linesf.astype(np.int32)
        colsf = colsf.astype(np.int32)
        sz = [n_lines, n_cols]

        inds_or1 = np.array([linesf, colsf, valuesf, sz])
        np.save('inds_or1', inds_or1)

        return inds_or1

    def get_wells(self):
        self.wells_n = self.mb.tag_get_data(self.wells_n_tag, 0, flat=True)[0]
        self.wells_d = self.mb.tag_get_data(self.wells_d_tag, 0, flat=True)[0]
        self.wells_n = self.mb.get_entities_by_handle(self.wells_n)
        self.wells_d = self.mb.get_entities_by_handle(self.wells_d)
        self.press = self.mb.tag_get_data(self.press_tag, self.wells_d, flat=True)
        self.vazao = self.mb.tag_get_data(self.q_tag, self.wells_n, flat=True)

    def kequiv(self,k1,k2):
        """
        obbtem o k equivalente entre k1 e k2

        """
        # keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def modificar_matriz(self, A, rows, columns, walk_rows, return_inds = False):
        """
        Modifica a matriz A para o tamanho (rows x columns)
        input:
            walk_rows: linhas para caminhar na matriz A
            rows: numero de linhas da nova matriz (C)
            columns: numero de colunas da nova matriz (C)
            return_inds: se return_inds = True retorna os indices das linhas, colunas
                         e respectivos valores
        output:
            C: CrsMatrix  rows x columns

        """
        lines = np.array([], dtype=np.int32)
        cols = lines.copy()
        valuesM = np.array([], dtype='float64')
        sz = [rows, columns]


        row_map = Epetra.Map(rows, 0, self.comm)
        col_map = Epetra.Map(columns, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 3)

        for i in range(walk_rows):
            p = A.ExtractGlobalRowCopy(i)
            values = p[0]
            index_columns = p[1]
            C.InsertGlobalValues(i, values, index_columns)
            lines = np.append(lines, np.repeat(i, len(values)))
            cols = np.append(cols, p[1])
            valuesM = np.append(valuesM, p[0])

        if return_inds == True:
            inds = [lines, cols, valuesM, sz]
            return C, inds
        else:
            return C

    def mod_transfine_wirebasket_by_inds(self, inds):
        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)

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

    def organize_op1(self):
        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat=True)
        map_gids = dict(zip(all_gids, self.all_volumes))
        ids_nv1 = self.mb.tag_get_data(self.L1_tag, self.all_volumes, flat=True)

        malha_fina = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBHEX, self.L3_tag,
            np.array([1]))

        max_ids_nv1 = max(ids_nv1)

        lines_op = np.array([])
        cols_op = np.array([])
        vals_op = np.array([], dtype=np.float64)

        my_primals = set()

        op_nv1 = np.load('inds_op.npy')
        for v in set(self.all_volumes) - set(malha_fina):
            id_vol_nv1 = self.mb.tag_get_data(self.L1_tag, v, flat=True)[0]
            gid_vol = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)[0]
            # print('gid_vol:{0}'.format(gid_vol))
            # print('id_vol_nv1:{0}'.format(id_vol_nv1))
            # print('\n')
            if id_vol_nv1 in my_primals:
                continue


            my_primals.add(id_vol_nv1)
            fine_to_primal_classic = self.mb.tag_get_data(self.fine_to_primal1_classic_tag, v, flat=True)[0]
            indices = np.where(op_nv1[1] == fine_to_primal_classic)[0]

            lines_op = np.append(lines_op, op_nv1[0][indices])
            cols_op = np.append(cols_op, np.repeat(id_vol_nv1, len(indices)))
            vals_op = np.append(vals_op, op_nv1[2][indices])



        for v in malha_fina:
            id_vol_nv1 = self.mb.tag_get_data(self.L1_tag, v, flat=True)[0]
            gid_vol = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)[0]
            # print('gid_vol:{0}'.format(gid_vol))
            # print('id_vol_nv1:{0}'.format(id_vol_nv1))
            # print('\n')
            # import pdb; pdb.set_trace()

            indices_line = np.where(lines_op == gid_vol)[0]

            if len(indices_line > 0):
                lines_op = np.delete(lines_op, indices_line)
                cols_op = np.delete(cols_op, indices_line)
                vals_op = np.delete(vals_op, indices_line)

            lines_op = np.append(lines_op, np.array([gid_vol]))
            cols_op = np.append(cols_op, np.array([id_vol_nv1]))
            vals_op = np.append(vals_op, np.array([1.0]))

        lines_op = lines_op.astype(np.int32)
        cols_op = cols_op.astype(np.int32)
        sz = [self.nf, max(ids_nv1)]

        inds_OP_ADM = np.array([lines_op, cols_op, vals_op, sz])
        # print(len(set(cols_op)))
        # print(set(cols_op))
        # print(my_primals)
        # import pdb; pdb.set_trace()

        # print(len(cols_op))
        # import pdb; pdb.set_trace()




        for i in set(cols_op):

            elems = self.mb.get_entities_by_type_and_tag(
                self.root_set, types.MBHEX, self.L1_tag,
                np.array([i]))
            gid_vol = self.mb.tag_get_data(self.global_id0_tag, elems, flat=True)
            op_adm_tag = self.mb.tag_get_handle(
                "OP_ADM{0}".format(i), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            indices = np.where(cols_op == i)[0]
            lines = lines_op[indices]
            vals = vals_op[indices]
            fine_elems = [map_gids[j] for j in lines]


            self.mb.tag_set_data(op_adm_tag, fine_elems, vals)

            # print(i)

            # print('gid_vol:{0}'.format(gid_vol))
            # print('id_vol_nv1:{0}'.format(id_vol_nv1))
            # print('\n')

            # print(i)
            # print(elems)
            # print('\n')

    def permutation_matrix(self):
        """
        G eh a matriz permutacao
        """
        self.map_global = dict(zip(self.all_volumes, range(self.nf)))

        global_map = list(range(self.nf))
        wirebasket_map = [self.map_global[i] for i in self.elems_wirebasket]
        global_map = np.array(global_map).astype(np.int32)
        wirebasket_map = np.array(wirebasket_map).astype(np.int32)

        std_map = Epetra.Map(self.nf, 0, self.comm)
        G = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)
        GT = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

        G.InsertGlobalValues(wirebasket_map, global_map, np.ones(self.nf, dtype=np.float64))

        return G, GT

    def put_indices_into_OP(self, inds, ind1, ind2):

        n_rows = inds[3][0]
        n_cols = inds[3][1]

        map_lines = dict(zip(range(n_rows), range(ind1, ind2)))

        lines = [map_lines[i] for i in inds[0]]
        cols = inds[1]
        values = inds[2]

        self.OP.InsertGlobalValues(lines, cols, values)

    def pymultimat(self, A, B, nf, transpose_A = False, transpose_B = False):
        """
        Multiplica a matriz A pela matriz B ambas de mesma ordem e quadradas
        nf: ordem da matriz

        """
        if A.Filled() == False:
            A.FillComplete()
        if B.Filled() == False:
            B.FillComplete()

        nf_map = Epetra.Map(nf, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, nf_map, 3)

        EpetraExt.Multiply(A, transpose_A, B, transpose_B, C)

        # C.FillComplete()

        return C

    def set_boundary(self, b, inds):
        """
        insere as condicoes de contorno na matriz de transmissibilidade
        da malha fina e no termo fonte
        input:
            b: termo fonte da gravidade
            inds: informacoes da matriz de transmissibilidade da malha fina

        output:
            inds2: informacoes da matriz de transmissiblidade da malha fina modificada
            b2: termo fonte modificado
        """
        inds2 = inds.copy()
        b2 = b

        wells_d = self.mb.tag_get_data(self.wells_d_tag, 0, flat=True)[0]
        wells_n = self.mb.tag_get_data(self.wells_n_tag, 0, flat=True)[0]
        wells_d = self.mb.get_entities_by_handle(wells_d)
        wells_n = self.mb.get_entities_by_handle(wells_n)
        # el = self.all_volumes[-1]
        # self.mb.tag_set_data(self.press_tag, el, 0.0)
        # wells_d = [el]
        # el = self.all_volumes[0]
        # self.mb.tag_set_data(self.q_tag, el, 1.0)
        # wells_n = [el]

        for v in wells_d:
            self.mb.tag_set_data(self.id_wells_tag, v, 1.0)
            gid = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)[0]
            indices = np.where(inds2[0] == gid)[0]
            inds2[0] = np.delete(inds2[0], indices)
            inds2[1] = np.delete(inds2[1], indices)
            inds2[2] = np.delete(inds2[2], indices)

            inds2[0] = np.append(inds2[0], np.array([gid]))
            inds2[1] = np.append(inds2[1], np.array([gid]))
            inds2[2] = np.append(inds2[2], np.array([1.0]))
            b2[gid] = self.mb.tag_get_data(self.press_tag, v, flat=True)[0]

        for v in wells_n:
            self.mb.tag_set_data(self.id_wells_tag, v, 1.0)
            gid = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)[0]
            b2[gid] += self.mb.tag_get_data(self.q_tag, v, flat=True)[0]

        return b2, inds2

    def set_global_problem_AMS_gr_faces(self, map_global):
        """
        transmissibilidade da malha fina
        input:
            map_global: mapeamento global
            return_inds: se return_inds == True, retorna o mapeamento da matriz sendo:
                         inds[0] = linhas
                         inds[1] = colunas
                         inds[2] = valores
                         inds[3] = tamanho da matriz trans_fine

        output:
            trans_fine: (multivector) transmissiblidade da malha fina
            b: (vector) termo fonte total
            s: (vector) termo fonte apenas da gravidade
            inds: mapeamento da matriz transfine
        obs: com funcao para obter dados dos elementos
        """
        #0
        nf = len(map_global)
        linesM = np.array([], dtype=np.int32)
        colsM = linesM.copy()
        valuesM = np.array([], dtype='float64')
        linesM2 = linesM.copy()
        valuesM2 = valuesM.copy()
        szM = [self.nf, self.nf]

        # lines = np.append(lines, np.repeat(i, len(values)))
        # cols = np.append(cols, p[1])
        # valuesM = np.append(valuesM, p[0])

        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set)

        std_map = Epetra.Map(self.nf, 0, self.comm)
        b = Epetra.Vector(std_map)
        s = Epetra.Vector(std_map)

        # cont = 0

        for face in set(self.all_faces) - set(all_faces_boundary_set):
            #1
            keq, s_grav, elems = self.get_kequiv_by_face_quad(face)

            linesM = np.append(linesM, [map_global[elems[0]], map_global[elems[1]]])
            colsM = np.append(colsM, [map_global[elems[1]], map_global[elems[0]]])
            valuesM = np.append(valuesM, [-keq, -keq])

            ind0 = np.where(linesM2 == map_global[elems[0]])
            if len(ind0[0]) == 0:
                linesM2 = np.append(linesM2, map_global[elems[0]])
                valuesM2 = np.append(valuesM2, [keq])
            else:
                valuesM2[ind0] += keq

            ind1 = np.where(linesM2 == map_global[elems[1]])
            if len(ind1[0]) == 0:
                linesM2 = np.append(linesM2, map_global[elems[1]])
                valuesM2 = np.append(valuesM2, [keq])
            else:
                valuesM2[ind1] += keq

            s[map_global[elems[0]]] += s_grav
            b[map_global[elems[0]]] += s_grav
            s[map_global[elems[1]]] += -s_grav
            b[map_global[elems[1]]] += -s_grav

        linesM = np.append(linesM, linesM2)
        colsM = np.append(colsM, linesM2)
        valuesM = np.append(valuesM, valuesM2)

        linesM = linesM.astype(np.int32)
        colsM = colsM.astype(np.int32)

        inds = np.array([linesM, colsM, valuesM, szM])


        return  b, s, inds

    def set_OP(self):
        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat=True)
        map_gids_in_volumes = dict(zip(all_gids, list(self.all_volumes)))
        self.inds_OP = np.load('inds_op.npy')

        OP = np.zeros((max(self.inds_OP[0])+1, max(self.inds_OP[0])+1), dtype=np.float64)
        OP[self.inds_OP[0], self.inds_OP[1]] = self.inds_OP[2]
        for primal in self.primals1:

            primal_id1 = self.mb.tag_get_data(self.prilmal_ids1_tag, primal, flat=True)[0]

            op_tag = self.mb.tag_get_handle(
                "OP_{0}".format(primal_id1), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)
            indice = np.nonzero(OP[:,primal_id1])[0]
            values = OP[indice, primal_id1]
            elems = [map_gids_in_volumes[i] for i in indice]
            self.mb.tag_set_data(op_tag, elems, values)

    def unitary(self,l):
        """
        obtem o vetor unitario positivo da direcao de l

        """
        uni = np.absolute(l/np.linalg.norm(l))
        # uni = np.abs(uni)

        return uni

    def run(self):
        map_global = dict(zip(self.all_volumes, range(self.nf)))

        b, s, inds = self.set_global_problem_AMS_gr_faces(self.map_wirebasket)
        inds_transmod = self.mod_transfine_wirebasket_by_inds(inds)
        # self.trans_mod = self.get_CrsMatrix_by_inds(inds_transmod)
        self.trans_mod = np.zeros((self.nf, self.nf), dtype=np.float64)
        self.trans_mod[inds_transmod[0], inds_transmod[1]] = inds_transmod[2]
        # self.G, self.GT = self.permutation_matrix()
        # self.get_OP()
        # self.set_OP()
        self.organize_op1()
        self.get_OR1()




        # bf, sf, indsf = self.set_global_problem_AMS_gr_faces(map_global)
        # std_map = Epetra.Map(self.nf, 0, self.comm)
        # bf = Epetra.Vector(std_map)
        # bf, indsf = self.set_boundary(bf, indsf)
        #
        # # for i in range(self.nf):
        # #     indices = np.where(indsf[0] == i)[0]
        # #     lines = indsf[0][indices]
        # #     cols = indsf[1][indices]
        # #     values = indsf[2][indices]
        # #     print(lines)
        # #     print(cols)
        # #     print(values)
        # #     print(sum(values))
        # #     print(bf[i])
        # #     print('\n')
        # #     import pdb; pdb.set_trace()
        #
        # A = self.get_CrsMatrix_by_inds(indsf)
        #
        # x = self.solve_linear_problem(A, bf, self.nf)
        # self.mb.tag_set_data(self.pf_tag, self.all_volumes, np.asarray(x))
        # import pdb; pdb.set_trace()


        self.mb.write_file('9x27x27_out_adm.vtk')

    def solve_linear_problem(self, A, b, n):

        if A.Filled():
            pass
        else:
            A.FillComplete()

        std_map = Epetra.Map(n, 0, self.comm)

        x = Epetra.Vector(std_map)

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(10000, 1e-14)

        return x

    def test_OP_tril(self, ind1 = None, ind2 = None):
        lim = 1e-7
        if ind1 == None and ind2 == None:
            verif = range(self.nf)
        elif ind1 == None or ind2 == None:
                print('defina ind1 e ind2')
                sys.exit(0)
        else:
            verif = range(ind1, ind2)

        for i in verif:
            p = self.OP.ExtractGlobalRowCopy(i)
            if sum(p[0]) > 1+lim or sum(p[0]) < 1-lim:
                print('Erro no Operador de Prologamento')
                print(i)
                print(sum(p[0]))
                import pdb; pdb.set_trace()



inputfile = '9x27x27.h5m'
sim = AMS_prol(inputfile)
