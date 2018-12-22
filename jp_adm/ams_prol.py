import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
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
principal = '/elliptic'
dir_output = '/elliptic/output'
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

        print('loading...')
        t1 = time.time()

        self.comm = Epetra.PyComm()
        self.mb = core.Core()
        self.mb.load_file(inputfile)
        self.mtu = topo_util.MeshTopoUtil(self.mb)
        self.root_set = self.mb.get_root_set()
        self.all_volumes = self.mb.get_entities_by_dimension(self.root_set, 3)

        self.all_faces = self.mb.get_entities_by_dimension(self.root_set, 2)
        self.all_edges = self.mb.get_entities_by_dimension(self.root_set, 1)
        self.nf = len(self.all_volumes)
        self.create_tags()
        self.get_wells()
        self.create_elems_wirebasket()
        self.map_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))

        self.primals1 = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.prilmal_ids1_tag]),
            np.array([None]))
        self.nc1 = len(self.primals1)
        self.primals2 = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.prilmal_ids2_tag]),
            np.array([None]))

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

        t2 = time.time()
        print('finish load')
        print('took:{0}\n'.format(t2-t1))


        # self.run()

    def calculo_pressao_corrigida(self, gids_adm_coarse_elems, all_faces_adm, intersect_faces_adm, all_faces_boundary_set, elems_nv0):

        for i in gids_adm_coarse_elems:
            linesM = np.array([])
            colsM = np.array([])
            valuesM = np.array([], dtype=np.float64)
            linesM2 = linesM.copy()
            valuesM2 = valuesM.copy()

            all_faces_coarse = all_faces_adm[i]
            boundary_faces_coarse = intersect_faces_adm[i]
            elems_in_primal = self.mb.get_entities_by_type_and_tag(
                self.root_set, types.MBHEX, self.L2_tag,
                np.array([i]))

            n = len(elems_in_primal)
            qpms_local = np.zeros(n)
            soma = 0.0
            szM = [n, n]
            map_local = dict(zip(elems_in_primal, range(n)))
            std_map = Epetra.Map(n, 0, self.comm)
            b = Epetra.Vector(std_map)

            for face in set(all_faces_coarse) - all_faces_boundary_set:
                keq, s_grav, elems = self.get_kequiv_by_face_quad(face)
                if face in boundary_faces_coarse:
                    pms = self.mb.tag_get_data(self.pms_tag, elems, flat=True)
                    # # sem gravidade
                    qpms = (pms[1] - pms[0])*keq
                    # # com gravidade
                    # qpms = (pms[1] - pms[0])*keq + s_grav
                    if elems[0] in elems_in_primal:
                        b[map_local[elems[0]]] += qpms
                        soma += qpms
                    else:
                        b[map_local[elems[1]]] -= qpms
                        soma -= qpms
                else:
                    linesM = np.append(linesM, [map_local[elems[0]], map_local[elems[1]]])
                    colsM = np.append(colsM, [map_local[elems[1]], map_local[elems[0]]])
                    valuesM = np.append(valuesM, [-keq, -keq])

                    ind0 = np.where(linesM2 == map_local[elems[0]])
                    if len(ind0[0]) == 0:
                        linesM2 = np.append(linesM2, map_local[elems[0]])
                        valuesM2 = np.append(valuesM2, [keq])
                    else:
                        valuesM2[ind0] += keq

                    ind1 = np.where(linesM2 == map_local[elems[1]])
                    if len(ind1[0]) == 0:
                        linesM2 = np.append(linesM2, map_local[elems[1]])
                        valuesM2 = np.append(valuesM2, [keq])
                    else:
                        valuesM2[ind1] += keq

                    # # com gravidade
                    # b[map_local[elems[0]]] += s_grav
                    # b[map_local[elems[1]]] -= s_grav

            lim = 1e-10
            if abs(soma) > lim:
                print('fluxo na malha grossa maior que zero')
                print(soma)
                import pdb; pdb.set_trace()


            linesM = np.append(linesM, linesM2)
            colsM = np.append(colsM, linesM2)
            valuesM = np.append(valuesM, valuesM2)

            # linesM = linesM.astype(np.int32)
            # colsM = colsM.astype(np.int32)

            inds2 = np.array([linesM, colsM, valuesM, szM])

            #elementos com pressao prescrita
            # elems_boundary_d = (set(self.wells_d) & set(elems_in_primal)) | (set(self.vertex_elems) & set(elems_in_primal))
            elems_boundary_d = set(self.vertex_elems) & set(elems_in_primal)

            for v in elems_boundary_d:
                id_local = map_local[v]
                indices = np.where(inds2[0] == id_local)[0]
                inds2[0] = np.delete(inds2[0], indices)
                inds2[1] = np.delete(inds2[1], indices)
                inds2[2] = np.delete(inds2[2], indices)

                inds2[0] = np.append(inds2[0], np.array([id_local]))
                inds2[1] = np.append(inds2[1], np.array([id_local]))
                inds2[2] = np.append(inds2[2], np.array([1.0]))
                b[id_local] = self.mb.tag_get_data(self.pms_tag, v, flat=True)[0]

            inds2[0] = inds2[0].astype(np.int32)
            inds2[1] = inds2[1].astype(np.int32)

            A = self.get_CrsMatrix_by_inds(inds2)
            x = self.solve_linear_problem(A, b, n)
            self.mb.tag_set_data(self.pcorr_tag, elems_in_primal, np.asarray(x))

            for face in set(all_faces_coarse) - all_faces_boundary_set:
                keq, s_grav, elems = self.get_kequiv_by_face_quad(face)
                if face in boundary_faces_coarse:
                    pms = self.mb.tag_get_data(self.pms_tag, elems, flat=True)
                    # # sem gravidade
                    qpms = (pms[1] - pms[0])*keq
                    # # com gravidade
                    # qpms = (pms[1] - pms[0])*keq + s_grav
                    if elems[0] in elems_in_primal:
                        qpms_local[map_local[elems[0]]] += qpms

                    else:
                        qpms_local[map_local[elems[1]]] -= qpms
                else:
                    pcorr = self.mb.tag_get_data(self.pcorr_tag, elems, flat=True)
                    # # sem gravidade
                    flux = (pcorr[1] - pcorr[0])*keq

                    # # com gravidade
                    # flux = (pcorr[1] - pcorr[0])*keq + s_grav

                    qpms_local[map_local[elems[0]]] += flux
                    qpms_local[map_local[elems[1]]] -= flux

            self.mb.tag_set_data(self.q_pms_tag, elems_in_primal, qpms_local)
            self.mb.tag_set_data(self.q_coarse_tag, elems_in_primal, np.repeat(soma, n))

        # para os elementos no nivel 0 a pressao corrigida eh a propria pms
        # para facilitar o carregamento se necessario
        pcorr = self.mb.tag_get_data(self.pms_tag, elems_nv0, flat=True)
        self.mb.tag_set_data(self.pcorr_tag, elems_nv0, pcorr)
        pcorr = self.mb.tag_get_data(self.pcorr_tag, self.all_volumes, flat=True)

        self.write_array('pcorr', np.asarray(pcorr))
        self.mb.tag_set_data(self.q_coarse_tag, elems_nv0, np.repeat(0.0, len(elems_nv0)))
        q_coarse = self.mb.tag_get_data(self.q_coarse_tag, self.all_volumes, flat=True)
        self.write_array('q_pms_coarse', q_coarse)

    def correcao_do_fluxo(self):
        """
        obtem a pressao corrigida
        """
        #todas as faces do contorno do dominio
        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set)

        gids_adm = self.mb.tag_get_data(self.L2_tag, self.all_volumes, flat=True)
        max_gids_adm = len(set(gids_adm))

        #todas as faces de cada volume da malha adm
        all_faces_adm = dict()

        for i in range(max_gids_adm):
            volumes = self.mb.get_entities_by_type_and_tag(
                self.root_set, types.MBHEX, self.L2_tag,
                np.array([i]))

            faces = self.mtu.get_bridge_adjacencies(volumes, 3, 2)

            all_faces_adm[i] = faces

        #faces de interseccao entre os volumes da malha adm
        intersect_faces_adm = dict()

        for i in range(max_gids_adm):
            for j in range(max_gids_adm):
                if i == j:
                    continue
                intersect = rng.intersect(all_faces_adm[i], all_faces_adm[j])
                if len(intersect) < 1:
                    continue
                try:
                    intersect_faces_adm[i] = rng.unite(intersect_faces_adm[i], intersect)
                except KeyError:
                    intersect_faces_adm[i] = intersect

        #########################################################
        #volumes no nivel 0
        elems_nv0 = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBHEX, self.L3_tag,
            np.array([1]))

        #todos os volumes que foram engrossados
        coarse_elems = rng.subtract(self.all_volumes, elems_nv0)
        gids_adm_coarse_elems = set(self.mb.tag_get_data(self.L2_tag, coarse_elems, flat=True))

        # self.set_q_pms_coarse(intersect_faces_adm, gids_adm_coarse_elems, elems_nv0, set(all_faces_boundary_set))

        #calculo da pressao corrigida
        self.calculo_pressao_corrigida(gids_adm_coarse_elems, all_faces_adm, intersect_faces_adm, set(all_faces_boundary_set), elems_nv0)

        # calculo do fluxo multiescala na malha fina
        self.fine_flux_pms(elems_nv0, set(all_faces_boundary_set))

    def create_elems_wirebasket(self):
        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat = True)

        map_global = dict(zip(self.all_volumes, all_gids))

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

        self.intern_elems = sorted(list(self.mb.get_entities_by_handle(intern_elems_meshset)), key=map_global.__getitem__)
        self.face_elems = sorted(list(self.mb.get_entities_by_handle(face_elems_meshset)), key=map_global.__getitem__)
        self.edge_elems = sorted(list(self.mb.get_entities_by_handle(edge_elems_meshset)), key=map_global.__getitem__)
        self.vertex_elems = sorted(list(self.mb.get_entities_by_handle(vertex_elems_meshset)), key=map_global.__getitem__)


        #####################################################################
        # all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat=True)
        # map_global = dict(zip(self.all_volumes, all_gids))
        #
        # self.intern_elems = set(self.mb.get_entities_by_type_and_tag(
        #     self.root_set, types.MBHEX, self.D1_tag, np.array([0])))
        #
        # self.face_elems = set(self.mb.get_entities_by_type_and_tag(
        #     self.root_set, types.MBHEX, self.D1_tag, np.array([1])))
        #
        # self.edge_elems = set(self.mb.get_entities_by_type_and_tag(
        #     self.root_set, types.MBHEX, self.D1_tag, np.array([2])))
        #
        # self.vertex_elems = set(self.mb.get_entities_by_type_and_tag(
        #     self.root_set, types.MBHEX, self.D1_tag, np.array([3])))
        #
        # self.intern_elems = sorted(list(self.intern_elems), key = map_global.__getitem__)
        # self.face_elems = sorted(list(self.face_elems), key = map_global.__getitem__)
        # self.edge_elems = sorted(list(self.edge_elems), key = map_global.__getitem__)
        # self.vertex_elems = sorted(list(self.vertex_elems), key = map_global.__getitem__)
        #####################################################################


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
        self.pms_tag = self.mb.tag_get_handle("PMS", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.erro_tag = self.mb.tag_get_handle("ERRO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.q_pf_tag = self.mb.tag_get_handle("Q_PF", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.q_pms_tag = self.mb.tag_get_handle("Q_PMS", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.pcorr_tag = self.mb.tag_get_handle("PCORR", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.s_grav_tag = self.mb.tag_get_handle("S_GRAV", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.keq_tag = self.mb.tag_get_handle("K_EQ", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.q_pms_coarse_tag = self.mb.tag_get_handle("Q_PMS_COARSE", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.q_coarse_tag = self.mb.tag_get_handle("Q_COARSE", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.vert_to_col_tag = self.mb.tag_get_handle("VERT_TO_COL", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        # self.id_wells_tag = self.mb.tag_get_handle("I", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    def delete_tag_op1(self):
        for i in range(self.nc1):
            tag = self.mb.tag_get_handle("OP")

    def fine_flux_pf(self):
        """
        fluxo da malha fina com solucao direta
        """

        print('getting fine flux pf')
        t0 = time.time()

        # import pdb; pdb.set_trace()

        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set)

        for face in set(self.all_faces) - set(all_faces_boundary_set):
            keq, s_grav, elems = self.get_kequiv_by_face_quad(face)

            try:
                q0 = self.mb.tag_get_data(self.q_pf_tag, elems[0], flat=True)[0]
            except RuntimeError:
                q0 = 0.0
                self.mb.tag_set_data(self.q_pf_tag, elems[0], q0)

            try:
                q1 = self.mb.tag_get_data(self.q_pf_tag, elems[1], flat=True)[0]
            except RuntimeError:
                q1 = 0.0
                self.mb.tag_set_data(self.q_pf_tag, elems[1], q1)



            pf = self.mb.tag_get_data(self.pf_tag, elems, flat=True)

            flux = (pf[1] - pf[0])*keq

            q0 += flux + s_grav
            q1 -= flux + s_grav

            self.mb.tag_set_data(self.q_pf_tag, elems, [q0, q1])

        t1 = time.time()
        fine_flux_pf = self.mb.tag_get_data(self.q_pf_tag, self.all_volumes, flat=True)
        self.write_array('fine_flux_pf', fine_flux_pf)

        print('took:{0}\n'.format(t1-t0))

    def fine_flux_pms(self, elems_nv0, all_faces_boundary_set):
        """
        fluxo da malha fina com solucao multiescala
        """
        #
        n = len(elems_nv0)
        qpms_fine = np.zeros(n)
        map_local = dict(zip(elems_nv0, range(n)))
        faces_elems_nv0 = self.mtu.get_bridge_adjacencies(elems_nv0, 3, 2)
        iter_faces = set(faces_elems_nv0) - (all_faces_boundary_set)

        for face in iter_faces:
            keq, s_grav, elems = self.get_kequiv_by_face_quad(face)
            pms = self.mb.tag_get_data(self.pms_tag, elems, flat=True)
            # # sem gravidade
            flux = (pms[1] - pms[0])*keq

            # # com gravidade
            # flux = (pms[1] - pms[0])*keq + s_grav

            if elems[0] in elems_nv0:
                qpms_fine[map_local[elems[0]]] += flux
            if elems[1] in elems_nv0:
                qpms_fine[map_local[elems[1]]] -= flux

        self.mb.tag_set_data(self.q_pms_tag, elems_nv0, qpms_fine)
        qpms_fine = self.mb.tag_get_data(self.q_pms_tag, self.all_volumes, flat=True)
        self.write_array('qpms_fine', qpms_fine)

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

        # self.inds_OP = np.array([np.array([]), np.array([]), np.array([],dtype = np.float64), [self.nf, self.nc1]])

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

        self.write_array('inds_op1', self.inds_OP)

        gids_vert = self.mb.tag_get_data(self.global_id0_tag, self.vertex_elems, flat=True)
        cols = []

        for i in gids_vert:
            indices = np.where(self.inds_OP[0] == i)[0]
            col = self.inds_OP[1][indices][0]
            cols.append(col)

        self.mb.tag_set_data(self.vert_to_col_tag, self.vertex_elems, cols)

        ###########################################################
        #debug

        # op = np.zeros((self.inds_OP[3][0], self.inds_OP[3][1]), dtype=np.float64)
        # op[self.inds_OP[0], self.inds_OP[1]] = self.inds_OP[2]
        #
        # cont = 0
        #
        # for i in op:
        #     ind = np.nonzero(i)[0]
        #     print(ind)
        #     print(cont)
        #     print(i[ind])
        #     print(sum(i[ind]))
        #
        #     print('\n')
        #     cont+=1
        #     import pdb; pdb.set_trace()
        ################################################

    def get_OR_ADM(self):

        linesf = np.array([])
        colsf = np.array([])
        valuesf = np.array([])

        gids_adm = self.mb.tag_get_data(self.L2_tag, self.all_volumes, flat = True)

        max_ids_adm = len(set(gids_adm))
        sz = [max_ids_adm, self.nf]

        for i in range(max_ids_adm):
            elems = self.mb.get_entities_by_type_and_tag(
                self.root_set, types.MBHEX, self.L2_tag,
                np.array([i]))
            gids_elems = self.mb.tag_get_data(self.global_id0_tag, elems, flat=True)

            colsf = np.append(colsf, gids_elems)
            linesf = np.append(linesf, np.repeat(i, len(gids_elems)))
            valuesf = np.append(valuesf, np.ones(len(gids_elems)))

            or_adm_tag = self.mb.tag_get_handle(
                "OR_ADM_{0}".format(i), 1, types.MB_TYPE_INTEGER, True,
                types.MB_TAG_SPARSE, default_value=0)
            self.mb.tag_set_data(or_adm_tag, elems, np.ones(len(elems), dtype=np.int))

        linesf = linesf.astype(np.int32)
        colsf = colsf.astype(np.int32)
        inds_or_adm = np.array([linesf, colsf, valuesf, sz])
        # np.save('inds_or_adm', inds_or_adm)
        self.write_array('inds_or_adm', inds_or_adm)

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

    @staticmethod
    def load_array(name):
        """
        carrega um array numpy da pasta output
        """

        # main_dir = '/elliptic'
        # out_dir = '/elliptic/output'

        main_dir = '/pytest'
        out_dir = '/pytest/output'

        os.chdir(out_dir)
        s = np.load(name)
        os.chdir(main_dir)
        return s

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

    def modificar_vetor(self, v, nc, n2=None):
        """
        Modifica o tamanho do vetor v para nc
        input:
            v:
                vetor a modificar o tamanho

            nc:
                tamanho requerido

            n2:
                linhas para igualar os valores do vetor (opcional)

        output:
            x:
                vetor com o tamanho modificado

        """

        if n2 == None:
            indices = np.arange(nc, dtype=np.int32)
        else:
            indices = np.arange(n2, dtype=np.int32)

        std_map = Epetra.Map(nc, 0, self.comm)
        x = Epetra.Vector(std_map)

        x[indices] = v[indices]

        return x

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

    def multimat_vector(self, A, row, b):
        """
        Multiplica a matriz A de ordem row x row pelo vetor de tamanho row

        """
        #0
        # if A.Filled() == False:
        #     A.FillComplete()


        std_map = Epetra.Map(row, 0, self.comm)

        c = Epetra.Vector(std_map)
        A.Multiply(False, b, c)

        return c

    def organize_op1(self):
        """
        obtem o operador de prolongamento adm
        """

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

        op_nv1 = self.load_array('inds_op1.npy')
        for v in set(self.vertex_elems):
            id_vol_nv1 = self.mb.tag_get_data(self.L1_tag, v, flat=True)[0]
            gid_vol = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)[0]
            col_op = self.mb.tag_get_data(self.vert_to_col_tag, v, flat=True)[0]

            indices = np.where(op_nv1[1] == col_op)[0]

            lines_op = np.append(lines_op, op_nv1[0][indices])
            cols_op = np.append(cols_op, np.repeat(id_vol_nv1, len(indices)))
            vals_op = np.append(vals_op, op_nv1[2][indices])

        for v in malha_fina:
            id_vol_nv1 = self.mb.tag_get_data(self.L1_tag, v, flat=True)[0]
            gid_vol = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)[0]

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
        sz = [self.nf, max_ids_nv1+1]

        inds_OP_ADM = np.array([lines_op, cols_op, vals_op, sz])
        self.write_array('inds_op_adm', inds_OP_ADM)

        for i in set(cols_op):

            elems = self.mb.get_entities_by_type_and_tag(
                self.root_set, types.MBHEX, self.L1_tag,
                np.array([i]))
            gid_vol = self.mb.tag_get_data(self.global_id0_tag, elems, flat=True)

            indices = np.where(cols_op == i)[0]
            if len(indices) < 2:
                continue
            lines = lines_op[indices]
            vals = vals_op[indices]
            fine_elems = [map_gids[j] for j in lines]

            op_adm_tag = self.mb.tag_get_handle(
                "OP_ADM{0}".format(i), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            self.mb.tag_set_data(op_adm_tag, fine_elems, vals)

    def permutation_matrix(self):
        """
        G eh a matriz permutacao
        """

        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat=True)
        self.map_global = dict(zip(self.all_volumes, all_gids))
        sz = [self.nf, self.nf]

        global_map = all_gids
        wirebasket_map = [self.map_global[i] for i in self.elems_wirebasket]
        global_map = np.array(global_map).astype(np.int32)
        wirebasket_map = np.array(wirebasket_map).astype(np.int32)

        std_map = Epetra.Map(self.nf, 0, self.comm)
        G = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)
        GT = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

        G.InsertGlobalValues(wirebasket_map, global_map, np.ones(self.nf, dtype=np.float64))
        GT.InsertGlobalValues(global_map, wirebasket_map, np.ones(self.nf, dtype=np.float64))
        inds_G = [wirebasket_map, global_map, np.ones(self.nf, dtype=np.float64), sz]
        inds_GT = [global_map, wirebasket_map, np.ones(self.nf, dtype=np.float64), sz]
        # np.save('inds_G', inds_G)
        self.write_array('inds_G', inds_G)
        # np.save('inds_GT', inds_GT)
        self.write_array('inds_GT', inds_GT)

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
            b:
                termo fonte da gravidade

            inds:
                informacoes da matriz de transmissibilidade da malha fina

        output:
            inds2:
                informacoes da matriz de transmissiblidade da malha fina com as condicoes de
                contorno

            b2:
                termo fonte com as condicoes de contorno
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
            gid = self.mb.tag_get_data(self.global_id0_tag, v, flat=True)[0]
            b2[gid] += self.mb.tag_get_data(self.q_tag, v, flat=True)[0]

        return b2, inds2

    def set_boundary_2(self, b, inds, map_global):
        """
        """
        n_elems_var = self.nf - len(self.wells_d)
        inds2 = inds.copy()
        inds2[3][0] = n_elems_var
        inds2[3][1] = n_elems_var
        elems_var = set(self.all_volumes) - set(self.wells_d)
        elems_var = sorted(list(elems_var), key = map_global.__getitem__)
        map_2 = dict
        std_map = Epetra.map(n_elems_var, 0, self.comm)
        b2 = Epetra.Vector(std_map)


        for v in self.wells_d:
            adjs = self.mtu.get_bridge_adjacencies(v, 2, 3)
            elems = [elem for elem in adjs if elem not in self.wells_d]
            press = self.mb.tag_get_data(self.press_tag, v, flat=True)[0]
            faces_v = self.mb.get_adjacencies(v, 2)

            for elem in elems:
                face_e = self.mb.get_adjacencies(elem, 2)
                face = list(set(faces_v) & set(face_e))
                face = face[0]
                keq, s_grav, elems = self.get_kequiv_by_face_quad(face)

        std_map = Epetra.Map(n_elems_var, 0, self.comm)
        b = Epetra.Vector(std_map)

    def set_erro(self):
        pf = self.load_array('pf.npy')
        pf += 1e-14
        pms = self.load_array('pms.npy')

        erro = 100*np.absolute(pf - pms)/pf

        self.mb.tag_set_data(self.erro_tag, self.all_volumes, erro)

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
        all_keqs = []
        all_faces_int = []
        all_s_grav = []
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
            all_keqs.append(keq)
            all_s_grav.append(s_grav)
            all_faces_int.append(face)

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

        self.mb.tag_set_data(self.keq_tag, all_faces_int, all_keqs)
        self.mb.tag_set_data(self.s_grav_tag, all_faces_int, all_s_grav)


        return  b, s, inds

    def set_global_problem_AMS_gr_faces_2(self, map_global):
        """
        transmissibilidade da malha fina excluindo os volumes de pressao prescrita
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
            if elems[0] in self.wells_d:
                press = self.mb.tag_get_data(self.press_tag, elems[0], flat=True)[0]
                gid0 = map_global[elems[0]]
                gid1 = map_global[elems[1]]

            elif elems[1] in self.wells_d:
                pass

            else:

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

        self.inds_OP = self.load_array('inds_op1.npy')

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

    def set_PC1(self):
        # pc = np.load('pc.npy')
        pc = self.load_array('pc.npy')

        for i in range(len(pc)):
            elems = self.mb.get_entities_by_type_and_tag(
                self.root_set, types.MBHEX, self.L1_tag,
                np.array([i]))
            pc_tag = self.mb.tag_get_handle(
                "PC_{0}".format(i), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)
            self.mb.tag_set_data(pc_tag, elems, np.repeat(pc[i], len(elems)))

    def set_Pcorr(self):
        # pms = np.load('pms.npy')
        pcorr = self.load_array('pcorr.npy')

        self.mb.tag_set_data(self.pcorr_tag, self.all_volumes, pcorr)

    def set_PMS(self):
        # pms = np.load('pms.npy')
        pms = self.load_array('pms.npy')

        self.mb.tag_set_data(self.pms_tag, self.all_volumes, pms)

    def set_PF(self):
        pf = self.load_array('pf.npy')

        self.mb.tag_set_data(self.pf_tag, self.all_volumes, pf)

    def set_QF(self):
        qf = self.load_array('fine_flux_pf.npy')
        self.mb.tag_set_data(self.q_pf_tag, self.all_volumes, qf)

    def set_QPms(self):
        qpms = self.load_array('q_pms_coarse.npy')
        self.mb.tag_set_data(self.q_pms_coarse_tag, self.all_volumes, qpms_coarse)

    def solucao_direta(self):

        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat = True)
        map_global = dict(zip(self.all_volumes, all_gids))
        print('trasnsfine')
        bf, sf, indsf = self.set_global_problem_AMS_gr_faces(map_global)
        print('set_boundary')

        std_map = Epetra.Map(self.nf, 0, self.comm)
        bf = Epetra.Vector(std_map)

        bf, indsf = self.set_boundary(bf, indsf)
        # np.save('b', bf)
        self.write_array('b', bf)
        # np.save('inds_transfine', indsf)
        self.write_array('inds_transfine', indsf)
        A = self.get_CrsMatrix_by_inds(indsf)
        print('solve')
        x = self.solve_linear_problem(A, bf, self.nf)
        # print('setting pf')
        # self.mb.tag_set_data(self.pf_tag, self.all_volumes, np.asarray(x))
        # np.save('pf', np.asarray(x))
        self.write_array('pf', np.asarray(x))
        # self.fine_flux_pf()

    def solucao_multiescala(self):

        print('solucao multiescala')

        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat = True)
        map_global = dict(zip(self.all_volumes, all_gids))

        n_adm = len(set(self.mb.tag_get_data(self.L2_tag, self.all_volumes, flat=True)))

        # op = np.load('inds_op_adm.npy')
        op = self.load_array('inds_op_adm.npy')
        sz = op[3][:]
        op[3] = [self.nf, self.nf]
        # or1 = np.load('inds_or1.npy')
        or1 = self.load_array('inds_or_adm.npy')
        or1[3] = [self.nf, self.nf]
        op = self.get_CrsMatrix_by_inds(op)
        or1 = self.get_CrsMatrix_by_inds(or1)
        bf, s, inds_transfine = self.set_global_problem_AMS_gr_faces(map_global)
        std_map = Epetra.Map(self.nf, 0, self.comm)
        bf = Epetra.Vector(std_map)
        bf, inds_transfine = self.set_boundary(bf, inds_transfine)

        transfine = self.get_CrsMatrix_by_inds(inds_transfine)

        tc = self.pymultimat(or1, transfine, self.nf)
        tc = self.pymultimat(tc, op, self.nf)
        tc, inds_tc = self.modificar_matriz(tc, sz[1], sz[1], sz[1], return_inds=True)
        # np.save('inds_tc', inds_tc)
        self.write_array('inds_tc', inds_tc)

        qc = self.multimat_vector(or1, self.nf, bf)
        qc = self.modificar_vetor(qc, sz[1])
        # np.save('qc', np.asarray(qc))
        self.write_array('qc', np.asarray(qc))
        pc = self.solve_linear_problem(tc, qc, sz[1])
        # np.save('pc', np.asarray(pc))
        self.write_array('pc', np.asarray(pc))
        # self.set_PC1()

        pc = self.modificar_vetor(pc, self.nf, n_adm)
        pms = self.multimat_vector(op, self.nf, pc)
        # np.save('pms', pms)
        self.write_array('pms', np.asarray(pms))
        # self.set_PMS()

        # print('correcao do fluxo')
        # t1 = time.time()
        # self.correcao_do_fluxo()
        # t2 = time.time()
        # print('took:{0}\n'.format(t2-t1))

    def solve_linear_problem(self, A, b, n):
        assert A.NumMyCols() == A.NumMyRows()
        assert A.NumMyCols() == len(b)

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

    def unitary(self,l):
        """
        obtem o vetor unitario positivo da direcao de l

        """
        uni = np.absolute(l/np.linalg.norm(l))
        # uni = np.abs(uni)

        return uni

    def run(self):

        self.verif = False
        all_gids = self.mb.tag_get_data(self.global_id0_tag, self.all_volumes, flat = True)
        map_global = dict(zip(self.all_volumes, all_gids))

        b, s, inds = self.set_global_problem_AMS_gr_faces(self.map_wirebasket)
        inds_transmod = self.mod_transfine_wirebasket_by_inds(inds)
        self.write_array('inds_transmod', inds_transmod)
        # # self.trans_mod = self.get_CrsMatrix_by_inds(inds_transmod)
        self.trans_mod = np.zeros((self.nf, self.nf), dtype=np.float64)
        self.trans_mod[inds_transmod[0], inds_transmod[1]] = inds_transmod[2]
        # self.G, self.GT = self.permutation_matrix()

        # print('get_op_ams')
        # t1 = time.time()
        # self.get_OP()
        # t2 = time.time()
        # print('took:{0}\n'.format(t2-t1))

        # print('setting_op_ams')
        # t1 = time.time()
        # self.set_OP()
        # t2 = time.time()
        # print('took:{0}\n'.format(t2-t1))

        # print('getting op_adm')
        # t1 = time.time()
        # self.organize_op1()
        # t2 = time.time()
        # print('took:{0}\n'.format(t2-t1))
        #
        print('getting or adm')
        t1 = time.time()
        self.get_OR_ADM()
        t2 = time.time()
        print('took:{0}\n'.format(t2-t1))

        # print('solucao_direta')
        # t1 = time.time()
        # self.solucao_direta()
        # t2 = time.time()
        # print('took:{0}\n'.format(t2-t1))
        #
        # print('solucao_multiescala')
        # t1 = time.time()
        # self.solucao_multiescala()
        # t2 = time.time()
        # print('took:{0}\n'.format(t2-t1))
        # #
        # print('erro')
        # t1 = time.time()
        # self.set_erro()
        # t2 = time.time()
        # print('took:{0}\n'.format(t2-t1))

        # self.set_PMS()
        # self.set_PF()
        # self.set_QF()





        # for i in range(self.nf):
        #     indices = np.where(indsf[0] == i)[0]
        #     lines = indsf[0][indices]
        #     cols = indsf[1][indices]
        #     values = indsf[2][indices]
        #     print(lines)
        #     print(cols)
        #     print(values)
        #     print(sum(values))
        #     print(bf[i])
        #     print('\n')
        #     import pdb; pdb.set_trace()

        # print('get_CrsMatrix_by_inds')

        # self.fine_flux_pf()
        # # import pdb; pdb.set_trace()
        ###################################################################

    def write_VTK(self, name):



        print('writting vtk file')
        t1 = time.time()
        self.mb.write_file(name)
        t2 = time.time()
        print('took:{0}\n'.format(t2-t1))

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

    @staticmethod
    def write_array(name, inf):
        """
        escreve arquivos do numpy na pasta output

        input:
            inf:
                array do numpy

            name:
                nome do arquivo
        """
        # main_dir = '/elliptic'
        # out_dir = '/elliptic/output'

        main_dir = '/pytest'
        out_dir = '/pytest/output'

        os.chdir(out_dir)
        np.save(name, inf)
        os.chdir(main_dir)


inputfile = '9x27x27.h5m'
sim = AMS_prol(inputfile)
sim.set_PMS()
sim.correcao_do_fluxo()

################################################
# rodar essa parte primeiro para obter as saidas
# sim.run()
# sim.solucao_direta()
# sim.solucao_multiescala()
#########################################################

##########################################
# escrever
sim.mb.delete_entities(sim.all_faces)
sim.mb.delete_entities(sim.all_edges)
name = '9x27x27_out_qpms_fine_adm.vtk'
sim.write_VTK(name)
##########################################

#########################################
# # setar as saidas

# sim.mb.delete_entities(sim.all_faces)
# sim.mb.delete_entities(sim.all_edges)
# sim.set_PF()
# sim.set_PMS()
# sim.set_erro()
# name = '9x27x27_out_adm.vtk'
# sim.write_VTK(name)
##############################################
