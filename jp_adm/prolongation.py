import numpy as np
from math import pi, sqrt
import time
import pyximport; pyximport.install()
import math
import os
import shutil
import random
import sys
import io
import yaml
from trilinos_utils import TrilinosUtils as triutils
from others_utils import OtherUtils as oth
import scipy.sparse as sp
from scipy.sparse.linalg import inv



class ProlongationTPFA3D:

    @staticmethod
    def get_tpfa_OP(comm, inds_mod, wirebasket_numbers):
        """
        obtem o operador de prolongamento wirebasket
        """

        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]

        idsi = ni
        idsf = idsi+nf
        idse = idsf+ne
        idsv = idse+nv
        loc = [idsi, idsf, idse, idsv]

        ntot = sum(wirebasket_numbers)

        OP = sp.lil_matrix((ntot, nv))
        t_mod = sp.lil_matrix((inds_mod[3][0], inds_mod[3][1]))
        t_mod[inds_mod[0], inds_mod[1]] = inds_mod[2]
        OP = ProlongationTPFA3D.insert_identity(OP, wirebasket_numbers)
        OP, inds_M = ProlongationTPFA3D.step1(comm, t_mod, OP, loc)
        OP, inds_M = ProlongationTPFA3D.step2(comm, t_mod, OP, loc, inds_M)
        OP = ProlongationTPFA3D.step3(comm, t_mod, OP, loc, inds_M)
        # OR = ProlongationTPFA3D.get_or(mb, OP)

        return OP

    @staticmethod
    def insert_identity(op, wirebasket_numbers):
        nv = wirebasket_numbers[3]
        nne = sum(wirebasket_numbers) - nv
        lines = np.arange(nne, nne+nv).astype(np.int32)
        values = np.ones(nv)
        matrix = sp.lil_matrix((nv, nv))
        rr = np.arange(nv).astype(np.int32)
        matrix[rr, rr] = values

        op[lines] = matrix

        return op

    @staticmethod
    def step1(comm, t_mod, op, loc):
        """
        elementos de aresta
        """
        lim = 1e-13

        nnf = loc[1]
        nne = loc[2]
        nnv = loc[3]
        ne = nne - nnf
        nv = nnv - nne
        nf = loc[1] - loc[0]
        M = t_mod[nnf:nne, nnf:nne]
        indices = M.nonzero()
        inds_M = [indices[0], indices[1], M[indices].toarray()[0], M.shape]
        inds_M = triutils.get_inverse_by_inds(comm, inds_M)
        M = triutils.get_CrsMatrix_by_inds(comm, inds_M)
        M2 = -1*t_mod[nnf:nne, nne:nnv]
        indices = M2.nonzero()
        inds_M2 = [indices[0], indices[1], M2[indices].toarray()[0], (ne, ne)]
        M2 = triutils.get_CrsMatrix_by_inds(comm, inds_M2)
        M = triutils.pymultimat(comm, M, M2)
        inds_M = triutils.get_inds_by_CrsMatrix(M)
        inds_M[3] = (ne, nv)
        matrix = sp.lil_matrix(inds_M[3])
        matrix[inds_M[0], inds_M[1]] = inds_M[2]

        op[nnf:nne] = matrix
        return op, inds_M

    @staticmethod
    def step2(comm, t_mod, op, loc, inds_MM):
        """
        elementos de face
        """
        nni = loc[0]
        nnf = loc[1]
        nne = loc[2]
        nnv = loc[3]
        ne = nne - nnf
        nv = nnv - nne
        nf = loc[1] - loc[0]
        ni = loc[0]

        if ne > nf:
            nt = ne
        else:
            nt = nf

        inds_MM[3] = (nt, nt)
        M = t_mod[nni:nnf, nni:nnf]
        # t0 = time.time()
        # Minv = inv(M.tocsc())
        # t1 = time.time()
        # print('tempo scipy')
        # print(t1-t0)
        # print('\n')
        indices = sp.find(M)
        inds_M = [indices[0], indices[1], indices[2], M.shape]
        # t0 = time.time()
        inds_M = triutils.get_inverse_by_inds(comm, inds_M)
        # t1 = time.time()
        # print('tempo trilinos')
        # print(t1-t0)
        # import pdb; pdb.set_trace()
        inds_M[3] = (nt, nt)
        M = triutils.get_CrsMatrix_by_inds(comm, inds_M) # nfxnf
        M2 = -1*t_mod[nni:nnf, nnf:nne] # nfxne
        indices = sp.find(M2)
        inds_M2 = [indices[0], indices[1], indices[2], (nt, nt)]
        M2 = triutils.get_CrsMatrix_by_inds(comm, inds_M2)
        M = triutils.pymultimat(comm, M, M2)
        MM = triutils.get_CrsMatrix_by_inds(comm, inds_MM)
        M = triutils.pymultimat(comm, M, MM)
        inds_M = triutils.get_inds_by_CrsMatrix(M)
        inds_M[3] = (nf, nv)
        matrix = sp.lil_matrix(inds_M[3])
        matrix[inds_M[0], inds_M[1]] = inds_M[2]

        op[nni:nnf] = matrix
        return op, inds_M

    @staticmethod
    def step3(comm, t_mod, op, loc, inds_MM):
        """
        elementos de face
        """
        nni = loc[0]
        nnf = loc[1]
        nne = loc[2]
        nnv = loc[3]
        ne = nne - nnf
        nv = nnv - nne
        nf = loc[1] - loc[0]
        ni = loc[0]

        nt = max([ni, nf, ne])

        inds_MM[3] = (nt, nt)
        M = t_mod[0:nni, 0:nni]
        indices = sp.find(M)
        inds_M = [indices[0], indices[1], indices[2], M.shape]
        inds_M = triutils.get_inverse_by_inds(comm, inds_M)
        inds_M[3] = (nt, nt)
        M = triutils.get_CrsMatrix_by_inds(comm, inds_M) # nfxnf
        M2 = -1*t_mod[0:nni, nni:nnf] # nfxne
        indices = sp.find(M2)
        inds_M2 = [indices[0], indices[1], indices[2], (nt, nt)]
        M2 = triutils.get_CrsMatrix_by_inds(comm, inds_M2)
        M = triutils.pymultimat(comm, M, M2)
        MM = triutils.get_CrsMatrix_by_inds(comm, inds_MM)
        M = triutils.pymultimat(comm, M, MM)
        inds_M = triutils.get_inds_by_CrsMatrix(M)
        inds_M[3] = (ni, nv)
        matrix = sp.lil_matrix(inds_M[3])
        matrix[inds_M[0], inds_M[1]] = inds_M[2]

        op[0:nni] = matrix
        return op
