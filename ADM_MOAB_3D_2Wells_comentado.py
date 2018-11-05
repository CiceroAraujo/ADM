import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
class MeshManager:
    def __init__(self,mesh_file, dim=3):
        self.dimension = dim
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)

        self.mb.load_file(mesh_file)

        self.physical_tag = self.mb.tag_get_handle("MATERIAL_SET")
        self.physical_sets = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, np.array(
            (self.physical_tag,)), np.array((None,)))

        self.dirichlet_tag = self.mb.tag_get_handle(
            "Dirichlet", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.neumann_tag = self.mb.tag_get_handle(
            "Neumann", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.perm_tag = self.mb.tag_get_handle(
            "Permeability", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.source_tag = self.mb.tag_get_handle(
            "Source term", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)


        self.all_volumes = self.mb.get_entities_by_dimension(0, self.dimension)

        self.all_nodes = self.mb.get_entities_by_dimension(0, 0)

        self.mtu.construct_aentities(self.all_nodes)
        self.all_faces = self.mb.get_entities_by_dimension(0, self.dimension-1)

        self.dirichlet_faces = set()
        self.neumann_faces = set()

        '''self.GLOBAL_ID_tag = self.mb.tag_get_handle(
            "Global_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True)'''

    def create_vertices(self, coords):
        new_vertices = self.mb.create_vertices(coords)
        self.all_nodes.append(new_vertices)
        return new_vertices

    def create_element(self, poly_type, vertices):
        new_volume = self.mb.create_element(poly_type, vertices)
        self.all_volumes.append(new_volume)
        return new_volume

    def set_information(self, information_name, physicals_values,
                        dim_target, set_connect=False):

        information_tag = self.mb.tag_get_handle(information_name)
        for physical, value in physicals_values.items():
            for a_set in self.physical_sets:
                physical_group = self.mb.tag_get_data(self.physical_tag,
                                                      a_set, flat=True)

                if physical_group == physical:
                    group_elements = self.mb.get_entities_by_dimension(a_set, dim_target)

                    if information_name == 'Dirichlet':
                        # print('DIR GROUP', len(group_elements), group_elements)
                        self.dirichlet_faces = self.dirichlet_faces | set(
                                                    group_elements)

                    if information_name == 'Neumann':
                        # print('NEU GROUP', len(group_elements), group_elements)
                        self.neumann_faces = self.neumann_faces | set(
                                                  group_elements)

                    for element in group_elements:
                        self.mb.tag_set_data(information_tag, element, value)

                        if set_connect:
                            connectivities = self.mtu.get_bridge_adjacencies(
                                                                element, 0, 0)
                            self.mb.tag_set_data(
                                information_tag, connectivities,
                                np.repeat(value, len(connectivities)))

    def get_boundary_nodes(self):
        all_faces = self.dirichlet_faces | self.neumann_faces
        boundary_nodes = set()
        for face in all_faces:
            nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
            boundary_nodes.update(nodes)
        return boundary_nodes

    def get_non_boundary_volumes(self, dirichlet_nodes, neumann_nodes):
        volumes = self.all_volumes
        non_boundary_volumes = []
        for volume in volumes:
            volume_nodes = set(self.mtu.get_bridge_adjacencies(volume, 0, 0))
            if (volume_nodes.intersection(dirichlet_nodes | neumann_nodes)) == set():
                non_boundary_volumes.append(volume)

        return non_boundary_volumes

    def set_media_property(self, property_name, physicals_values,
                           dim_target=3, set_nodes=False):

        self.set_information(property_name, physicals_values,
                             dim_target, set_connect=set_nodes)

    def set_boundary_condition(self, boundary_condition, physicals_values,
                               dim_target=3, set_nodes=False):

        self.set_information(boundary_condition, physicals_values,
                             dim_target, set_connect=set_nodes)

    def get_centroid(self, entity):

        verts = mbcore.get_connectivity(entity)
        coords = np.array([self.mb.get_coords([vert]) for vert in verts])

        qtd_pts = len(verts)
        #print qtd_pts, 'qtd_pts'
        coords = np.reshape(coords, (qtd_pts, 3))
        pseudo_cent = sum(coords)/qtd_pts
        return pseudo_cent

    def get_tetra_volume(self, tet_nodes):
        vect_1 = tet_nodes[1] - tet_nodes[0]
        vect_2 = tet_nodes[2] - tet_nodes[0]
        vect_3 = tet_nodes[3] - tet_nodes[0]
        vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3))/6.0
        return vol_eval

    @staticmethod
    def point_distance(coords_1, coords_2):
        dist_vector = coords_1 - coords_2
        distance = sqrt(np.dot(dist_vector, dist_vector))
        return distance

M1= MeshManager('16.msh')          # Objeto que armazenará as informações da malha
all_volumes=M1.all_volumes

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)
C1=2
C2=2

# Posição aproximada de cada completação
Cent_weels=[[0.5, 0.5, 0.5], [7.5, 0.5, 7.5]]

# Distância, em relação ao poço, até onde se usa malha fina
r0=0.4

# Distância, em relação ao poço, até onde se usa malha intermediária (Ainda não implementado)
r1=2

#--------------fim dos parâmetros de entrada------------------------------------

#--------------Definição das dimensões dos elementos da malha fina--------------
# Esse bloco deve ser alterado para uso de malhas não estruturadas
verts = M1.mb.get_connectivity(all_volumes[0])     #Vértices de um elemento da malha fina
coords = np.array([M1.mb.get_coords([vert]) for vert in verts])
xmin, xmax = coords[0][0], coords[0][0]
ymin, ymax = coords[0][1], coords[0][1]
zmin, zmax = coords[0][2], coords[0][2]
for c in coords:
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[2]
    if c[2]<zmin: zmin=c[2]
dx0, dy0, dz0 = xmax-xmin, ymax-ymin, zmax-zmin # Tamanho de cada elemento na malha fina
#-------------------------------------------------------------------------------

# Definição dos volumes que pertencem à malha fina e armazenamento em uma lista
wells=[]
G_ID_min=0              #É usado para determinar o manor dos global IDs de volume
for e in all_volumes:
    e_tags=M1.mb.tag_get_tags_on_entity(e)
    if M1.mb.tag_get_data(e_tags[0], e, flat=True)>G_ID_min:
        G_ID_min=M1.mb.tag_get_data(e_tags[0], e, flat=True)
    centroid=M1.mtu.get_average_position([e])    
    for c in Cent_weels:
        dx=(centroid[0]-c[0])**2
        dy=(centroid[1]-c[1])**2
        dz=(centroid[2]-c[2])**2
        if dx<r0**2 and dy<r0**2 and dz<r0**2:
            wells.append(e)
#-------------------------------------------------------------------------------

#Determinação das dimensões do reservatório e dos volumes das malhas intermediária e grossa
for v in M1.all_nodes:       # M1.all_nodes -> todos os vértices da malha fina
    c=M1.mb.get_coords([v])  # Coordenadas de um nó
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[0]
    if c[2]<zmin: zmin=c[2]

Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin  # Dimensões do reservatório
dx1, dy1, dz1 = C1*dx0, C1*dy0, C1*dz0        # Dimensões dos volumes da malha intermediária
dx2, dy2, dz2 = C2*dx1, C2*dy1, C2*dz1        # Dimensões dos volumes da malha grossa
#-------------------------------------------------------------------------------

# Criação do vetor que define a "grade" que separa os volumes da malha grossa
# Essa grade é absoluta (relativa ao reservatório como um todo)
lx2, ly2, lz2 = [], [], []
for i in range(int(Lx/(C1*C2*dx0))+1):    lx2.append(xmin+i*dx2)
for i in range(int(Ly/(C1*C2*dy0))+1):    ly2.append(ymin+i*dy2)
for i in range(int(Lz/(C1*C2*dz0))+1):    lz2.append(zmin+i*dz2)
#-------------------------------------------------------------------------------

# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(C2):    lx1.append(i*dx1)
for i in range(C2):    ly1.append(i*dy1)
for i in range(C2):    lz1.append(i*dz1)
#-------------------------------------------------------------------------------
print(lx2,lx1)
t0=time.time()
# ---- Criação e preenchimento da árvore de meshsets----------------------------
# Esse bloco é executado apenas uma vez em um problema bifásico, sua eficiência
# não é criticamente importante.
L2_meshset=M1.mb.create_meshset()       # root Meshset
print(all_volumes[0])
for i in lx2:
    for j in ly2:
        for k in lz2:
            l2_meshset=M1.mb.create_meshset()
            for elem in all_volumes:
                # Refactory here
                # Verificar se o uso de um vértice reduz o custo
                centroid=M1.mtu.get_average_position([elem])
                if (centroid[0]>i) and (centroid[0]<i+dx2) and (centroid[1]>j)\
                and (centroid[1]<j+dy2) and (centroid[2]>k) and (centroid[2]<k+dz2):
                    M1.mb.add_entities(l2_meshset, [elem])
                    elem_por_L2=M1.mb.get_entities_by_handle(l2_meshset)
            for m in lx1:
                for n in ly1:
                    for o in lz1:
                        l1_meshset=M1.mb.create_meshset()
                        for e in elem_por_L2:
                            # Refactory here
                            # Verificar se o uso de um vértice reduz o custo
                            centroid=M1.mtu.get_average_position([e])
                            if (centroid[0]>i+m) and (centroid[0]<i+m+dx1)\
                            and (centroid[1]>j+n) and (centroid[1]<j+n+dy1)\
                            and (centroid[2]>k+o) and (centroid[2]<k+o+dz1):
                                M1.mb.add_entities(l1_meshset,[e])
                                M1.mb.add_child_meshset(l2_meshset,l1_meshset)
                                M1.mb.add_child_meshset(L2_meshset,l2_meshset)
                        elem_por_L1=M1.mb.get_entities_by_handle(l1_meshset)
#-------------------------------------------------------------------------------
print('Criação da árvore: ',time.time()-t0)
t0=time.time()
# --------------Atribuição dos IDs de cada nível em cada volume-----------------
# Esse bloco é executado uma vez a cada iteração em um problema bifásico,
# sua eficiência é criticamente importante.

# Tag que armazena o ID do volume no nível 1
L1_ID_tag=M1.mb.tag_get_handle("l1_ID", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_DENSE, True)
# Tag que armazena o ID do volume no nível 2
L2_ID_tag=M1.mb.tag_get_handle("l2_ID", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_DENSE, True)
# ni = ID do elemento no nível i
n1=0
n2=0
aux=0
meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1=M1.mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = M1.mb.get_entities_by_handle(m1)
        for elem1 in elem_by_L1:
            elem1_tags = M1.mb.tag_get_tags_on_entity(elem1)
            elem1_Global_ID = M1.mb.tag_get_data(elem1_tags[0], elem1, flat=True)
            if elem1 in wells:
                aux=1
                tem_poço_no_vizinho=True
        if aux==1:
            aux=0
            for elem in elem_by_L1:
                n1+=1
                n2+=1
                M1.mb.tag_set_data(L1_ID_tag, elem, np.array([n1], dtype=np.float))
                M1.mb.tag_set_data(L2_ID_tag, elem, np.array([n2], dtype=np.float))
                elem_tags = M1.mb.tag_get_tags_on_entity(elem)
                elem_Global_ID = M1.mb.tag_get_data(elem_tags[0], elem, flat=True)
                wells.append(elem)
    if tem_poço_no_vizinho:
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            n1+=1
            n2+=1
            t=1
            for elem in elem_by_L1:
                if elem not in wells:
                    M1.mb.tag_set_data(L1_ID_tag, elem, np.array([n1], dtype=np.float))
                    M1.mb.tag_set_data(L2_ID_tag, elem, np.array([n2], dtype=np.float))
                    t=0
            n1-=t
            n2-=t
    else:
        n2+=1
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            n1+=1
            for elem2 in elem_by_L1:
                elem2_tags = M1.mb.tag_get_tags_on_entity(elem)
                M1.mb.tag_set_data(L2_ID_tag, elem2, np.array([n2], dtype=np.float))
                M1.mb.tag_set_data(L1_ID_tag, elem2, np.array([n1], dtype=np.float))
# ------------------------------------------------------------------------------
print('Distribuição das tags: ',time.time()-t0)
t0=time.time()
# Criação e preenchimento do operador de restrição do nível 0 para o nível 1
R01=np.zeros((n1,len(all_volumes)),dtype=np.float)
for e in all_volumes:
    elem_tags = M1.mb.tag_get_tags_on_entity(e)
    elem_Global_ID = int(M1.mb.tag_get_data(elem_tags[0], e, flat=True))
    elem_ID1 = int(M1.mb.tag_get_data(elem_tags[1], e, flat=True))
    R01[elem_ID1-1][elem_Global_ID-G_ID_min]=1
# ------------------------------------------------------------------------------

# Criação e preenchimento do operador de restrição do nível 1 para o nível 2
R12=np.zeros((n2,n1),dtype=np.float)
for e in all_volumes:
    elem_tags = M1.mb.tag_get_tags_on_entity(e)
    elem_ID1 = int(M1.mb.tag_get_data(elem_tags[1], e, flat=True))
    elem_ID2 = int(M1.mb.tag_get_data(elem_tags[2], e, flat=True))
    R12[elem_ID2-1][elem_ID1-1]=1
# ------------------------------------------------------------------------------
print('Criação dos operadores: ',time.time()-t0)
print(n1,n2)
M1.mb.write_file('teste_3D.vtk')
print('New file created')
