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
import tools_trilinos as totr

def mount_linear_system_upscaling_flow_channel(infos):
    """
    """
    comm = infos['comm']
    mb = infos['mb']
    mtu = infos['mtu']
    volumes = np.array(infos['elems'])
    elems_min = infos['elems_min']
    elems_max = infos['elems_max']
    keq_tag = infos['keq_tag']
    cent_tag = infos['cent_tag']
    area_tag = infos['area_tag']
    mi = 1.0

    lines = np.array()
    cols = lines.copy()
    values = lines.copy()
    linesM2 = linesM.copy()
    valuesM2 = valuesM.copy()
    n = len(elems)
    szM = [n, n]

    map_local = dict(zip(volumes, range(n)))
    std_map = Epetra.Map(self.nf, 0, self.comm)
    b = Epetra.Vector(std_map)


    #montando a transmissibilidade
    faces = mtu.get_bridge_adjacencies(elems, 3, 2)
    all_keqs = mb.tag_get_data(keq_tag, faces, flat=True)
    map_keq = dict(zip(faces, all_keqs))

    KEQ = []
    P0 = 1.0
    P1 = 0.0
    DP = P0 - P1
    faces_in = set()

    for face in faces:
        elems = self.mb.get_adjacencies(face, 3)
        if len(elems) < 2:
            continue
        if elems[0] not in volumes or elems[1] not in volumes:
            continue
        keq = map_keq[face]
        faces_in.add(face)
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

    linesM = np.append(linesM, linesM2)
    colsM = np.append(colsM, linesM2)
    valuesM = np.append(valuesM, valuesM2)

    linesM = linesM.astype(np.uint32)
    colsM = colsM.astype(np.uint32)

    inds_transfine_local = np.array([linesM, colsM, valuesM, szM, [], []])
    for i in range(3):
        inds_transfine, b = set_boundary_dirichlet(map_local, elems_min[i], np.repeat(P0, len(elems_min[i])), b, inds_transfine_local)
        inds_transfine, b = set_boundary_dirichlet(map_local, elems_max[i], np.repeat(P1, len(elems_max[i])), b, inds_transfine_local)
        A = get_CrsMatrix_by_inds(inds_transfine, comm)
        x = totr.solve_linear_problem(comm,A,b)
        cent_min = mb.tag_get_data(cent_tag, elems_min[i])[:,i].min()
        cent_max = mb.tag_get_data(cent_tag, elems_max[i])[:,i].max()
        D = cent_max - cent_min
        faces_max = set(mtu.get_bridge_adjacencies(elems_max, 3, 2)) & faces_in
        flux = {}
        for face in faces_max:
            keq = map_keq[face]
            elems = self.mb.get_adjacencies(face, 3)
            flux.setdefault(elems[0],0.0)
            flux.setdefault(elems[1],0.0)
            pf = x[map_local[elems[0]], map_local[elems[1]]]
            f = (pf[1] - pf[0])*keq
            flux[elems[0]] += f
            flux[elems[1]] -= f

        qmax = sum(flux.values())
        faces_sec = []
        faces_max = set(mtu.get_bridge_adjacencies(elems_max, 3, 2))-faces_in
        for face in faces_max:
            elems = self.mb.get_adjacencies(face, 3)
            if len(elems) < 2:
                continue
            faces_sec.append(face)
        sum_areas = mb.tag_get_data(area_tag, faces_sec, flat=True).sum()
        keq_i = abs((qmax*mi*D)/(sum_areas * DP))
        KEQ.append(keq_i)

    return KEQ

def set_boundary_dirichlet(map_local, boundary_elems, values, b, inds):
    map_values = dict(zip(boundary_elems, values))
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

    return inds2, b

def get_CrsMatrix_by_inds(inds, comm):
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

    if len(inds[4]) < 1:
        A.InsertGlobalValues(inds[0], inds[1], inds[2])
    else:
        A.InsertGlobalValues(inds[4], inds[5], inds[2])
    # else:
    #     raise ValueError("especifique true ou false para slice")

    return A
