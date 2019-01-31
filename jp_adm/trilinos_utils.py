import numpy as np
from math import pi, sqrt
# from pymoab import core, types, rng, topo_util, skinner
import time
import pyximport; pyximport.install()
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import math
import os
import shutil
import random
import sys
# import configparser
# import io
# import yaml



class TrilinosUtils:

    @staticmethod
    def solve_linear_problem(comm, A, b):
        """
        retorna a solucao do sistema linear Ax = b
        input:
            A: matriz do sistema
            b: termo fonte
        output:
            x:solucao
        """

        n = len(b)
        assert A.NumMyCols() == A.NumMyRows()
        assert A.NumMyCols() == n

        if A.Filled():
            pass
        else:
            A.FillComplete()

        std_map = Epetra.Map(n, 0, comm)

        x = Epetra.Vector(std_map)

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(10000, 1e-14)

        return x

    @staticmethod
    def get_CrsMatrix_by_inds(comm, inds):
        """
        retorna uma CrsMatrix a partir de inds
        input:
            inds: array numpy com informacoes da matriz
        output:
            A: CrsMatrix
        """

        rows = inds[3][0]
        cols = inds[3][1]

        row_map = Epetra.Map(rows, 0, comm)
        col_map = Epetra.Map(cols, 0, comm)
        A = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 7)
        A.InsertGlobalValues(inds[0], inds[1], inds[2])

        return A

    @staticmethod
    def get_inverse_tril(comm, A):
        """
        Obter a matriz inversa de A
        obs: A deve ser quadrada
        input:
            A: CrsMatrix
        output:
            Inv: CrsMatrix inversa de A
        """
        num_cols = A.NumMyCols()
        num_rows = A.NumMyRows()
        assert num_cols == num_rows
        map1 = Epetra.Map(rows, 0, comm)

        Inv = Epetra.CrsMatrix(Epetra.Copy, map1, 3)
        lines2 = np.array([])
        cols2 = np.array([])
        values2 = np.array([])

        for i in range(num_rows):
            b = Epetra.Vector(map1)
            b[i] = 1.0

            x = TrilinosUtils.solve_linear_problem(comm, A, b)
            lines = np.nonzero(x[:])[0].astype(np.int32)
            col = np.repeat(i, len(lines)).astype(np.int32)
            lines2 = np.append(lines2, lines)
            cols2 = np.append(cols2, col)
            values2 = np.append(values2, x[lines])

        lines2 = lines2.astype(np.int32)
        cols2 = cols2.astype(np.int32)

        Inv.InsertGlobalValues(lines2, cols2, values2)

        return Inv

    @staticmethod
    def get_inverse_by_inds(comm, inds):
        """
        retorna inds da matriz inversa a partir das informacoes (inds) da matriz de entrada
        """

        assert inds[3][0] == inds[3][1]
        cols = inds[3][1]
        sz = [cols, cols]
        A = TrilinosUtils.get_CrsMatrix_by_inds(comm, inds)

        lines2 = np.array([])
        cols2 = np.array([])
        values2 = np.array([], dtype=np.float64)
        map1 = Epetra.Map(cols, 0, comm)

        for i in range(cols):
            b = Epetra.Vector(map1)
            b[i] = 1.0

            x = TrilinosUtils.solve_linear_problem(comm, A, b)

            lines = np.nonzero(x[:])[0]
            col = np.repeat(i, len(lines))
            vals = x[lines]

            lines2 = np.append(lines2, lines)
            cols2 = np.append(cols2, col)
            values2 = np.append(values2, vals)

        lines2 = lines2.astype(np.int32)
        cols2 = cols2.astype(np.int32)

        inds2 = np.array([lines2, cols2, values2, sz, False, False])

        return inds2

    @staticmethod
    def pymultimat(comm, A, B, transpose_A=False, transpose_B=False):
        """
        Multiplica a matriz A pela matriz B ambas de mesma ordem e quadradas
        nf: ordem da matriz

        """
        assert A.NumMyCols() == A.NumMyRows()
        assert B.NumMyCols() == B.NumMyRows()
        assert A.NumMyRows() == B.NumMyRows()
        n = A.NumMyCols()

        if A.Filled() == False:
            A.FillComplete()
        if B.Filled() == False:
            B.FillComplete()

        nf_map = Epetra.Map(n, 0, comm)

        C = Epetra.CrsMatrix(Epetra.Copy, nf_map, 3)

        EpetraExt.Multiply(A, transpose_A, B, transpose_B, C)

        return C

    @staticmethod
    def get_inds_by_CrsMatrix(A):
        """
        retorna os indices da matriz do trilinos
        """

        lines = np.array([], dtype=np.int32)
        cols = lines.copy()
        valuesM = np.array([], dtype='float64')
        rows = A.NumMyRows()
        columns = A.NumMyCols()
        sz = (rows, columns)

        for i in range(rows):
            p = A.ExtractGlobalRowCopy(i)
            values = p[0]
            index_columns = p[1]
            if len(p[1]) > 0:
                lines = np.append(lines, np.repeat(i, len(values)))
                cols = np.append(cols, p[1])
                valuesM = np.append(valuesM, p[0])

        lines = lines.astype(np.int32)
        cols = cols.astype(np.int32)

        inds = np.array([lines, cols, valuesM, sz])
        return inds
