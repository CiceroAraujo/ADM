import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
import time
# import pyximport; pyximport.install()
# from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import math
import os
import shutil
import random
import sys
# import configparser
import io
import yaml


class UtilsPymoab:

    @staticmethod
    def get_faces(mb, elements):
        mtu = topo_util.MeshTopoUtil(mb)
        elements = UtilsPymoab.get_elements(mb, elements)
        faces = mtu.get_bridge_adjacencies(elements, 3, 2)
        return faces

    @staticmethod
    def get_boundary_of_volumes(mb, elements):
        """
        input:
            mb: core of pymoab
            elements: meshset or elements of the mesh
        output:
            boundary of meshset or elements
        """
        faces = UtilsPymoab.get_faces(mb, elements)

        bound_faces = []

        for face in faces:
            elems = mb.get_adjacencies(face, 3)
            if len(elems) < 2:
                bound_faces.append(face)
            elif elems[0] in elements and elems[1] not in elements:
                bound_faces.append(face)
            elif elems[1] in elements and elems[0] not in elements:
                bound_faces.append(face)

        return rng.Range(bound_faces)

    @staticmethod
    def get_faces_in_intersection_between_volumes(mb, elements1, elements2):
        """
        Retorna as faces na interseccao entre dois ranges de volumes ou dois meshsets
        """
        bound_faces1 = UtilsPymoab.get_boundary_of_volumes(mb, elements1)
        bound_faces2 = UtilsPymoab.get_boundary_of_volumes(mb, elements2)
        return rng.intersect(bound_faces1, bound_faces2)

    @staticmethod
    def get_elements(mb, elements):
        """
        retorna o rng.Range dos elementos de entrada
        """
        if isinstance(elements, rng.Range):
            return elements
        elif isinstance(elements, int):
            elements = mb.get_entities_by_handle(elements)
            return elements
        elif isinstance(elements, np.ndarray) or isinstance(elements, list):
            return rng.Range(elements)
        else:
            raise ValueError('tipo de dado incorreto')

    @staticmethod
    def get_all_entities(mb):
        mtu = topo_util.MeshTopoUtil(mb)
        root_set = mb.get_root_set()
        all_volumes = mb.get_entities_by_dimension(0, 3)
        all_nodes = mb.get_entities_by_dimension(0, 0)
        mtu.construct_aentities(all_nodes)
        all_faces = mb.get_entities_by_dimension(0, 2)
        all_edges = mb.get_entities_by_dimension(0, 1)

        return [all_nodes, all_edges, all_faces, all_volumes]
