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
import io
import yaml



def solve_linear_problem(comm, A, b):
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
