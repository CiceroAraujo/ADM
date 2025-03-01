import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os

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

        #self.perm_tag = self.mb.tag_get_handle(
        #    "Permeability", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

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

        self.create_tags()
        self.set_k()
        #self.set_information("PERM", self.all_volumes, 3)
        #self.get_boundary_faces()
        self.gravity = False
        self.gama = 10

    def create_tags(self):
        self.perm_tag = self.mb.tag_get_handle("PERM", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.finos_tag = self.mb.tag_get_handle("finos", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_dirichlet_tag = self.mb.tag_get_handle("WELLS_D", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_neumann_tag = self.mb.tag_get_handle("WELLS_N", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.press_value_tag = self.mb.tag_get_handle("P", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.vazao_value_tag = self.mb.tag_get_handle("Q", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.all_faces_boundary_tag = self.mb.tag_get_handle("FACES_BOUNDARY", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.area_tag = self.mb.tag_get_handle("AREA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.GLOBAL_ID_tag = self.mb.tag_get_handle("G_ID_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.ID_reordenado_tag = self.mb.tag_get_handle("ID_reord_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

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
        for physical_value in physicals_values:
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

    def set_k(self):
        k = 1.0
        perm_tensor = [k, 0, 0,
                       0, k, 0,
                       0, 0, k]
        for v in self.all_volumes:
            self.mb.tag_set_data(self.perm_tag, v, perm_tensor)
            #v_tags=self.mb.tag_get_tags_on_entity(v)
            #print(self.mb.tag_get_data(v_tags[1],v,flat=True))

    def set_area(self, face):


        points = self.mtu.get_bridge_adjacencies(face, 2, 0)
        points = [self.mb.get_coords([vert]) for vert in points]
        if len(points) == 3:
            n1 = np.array(points[0] - points[1])
            n2 = np.array(points[0] - points[2])
            area = (np.linalg.norm(np.cross(n1, n2)))/2.0

        #calculo da area para quadrilatero regular
        elif len(points) == 4:
            n = np.array([np.array(points[0] - points[1]), np.array(points[0] - points[2]), np.array(points[0] - points[3])])
            norms = np.array(list(map(np.linalg.norm, n)))
            ind_norm_max = np.where(norms == max(norms))[0]
            n = np.delete(n, ind_norm_max, axis = 0)

            area = np.linalg.norm(np.cross(n[0], n[1]))

        self.mb.tag_set_data(self.area_tag, face, area)

    def get_boundary_nodes(self):
        all_faces = self.dirichlet_faces | self.neumann_faces
        boundary_nodes = set()
        for face in all_faces:
            nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
            boundary_nodes.update(nodes)
        return boundary_nodes

    def get_faces_boundary(self):
        """
        cria os meshsets
        all_faces_set: todas as faces do dominio
        all_faces_boundary_set: todas as faces no contorno
        """
        all_faces_boundary_set = self.mb.create_meshset()

        for face in self.all_faces:
            size = len(self.mb.get_adjacencies(face, 3))
            self.set_area(face)
            if size < 2:
                self.mb.add_entities(all_faces_boundary_set, [face])

        self.mb.tag_set_data(self.all_faces_boundary_tag, 0, all_faces_boundary_set)

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

    def get_tetra_volume(self, tet_nodes):
        vect_1 = tet_nodes[1] - tet_nodes[0]
        vect_2 = tet_nodes[2] - tet_nodes[0]
        vect_3 = tet_nodes[3] - tet_nodes[0]
        vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3))/6.0
        return vol_eval

    def get_boundary_faces(self):
        all_boundary_faces = self.mb.create_meshset()
        for face in self.all_faces:
            elems = self.mtu.get_bridge_adjacencies(face, 2, 3)
            if len(elems) < 2:
                self.mb.add_entities(all_boundary_faces, [face])

        self.mb.tag_set_data(self.all_faces_boundary_tag, 0, all_boundary_faces)

    @staticmethod
    def imprima(self, text = None):
        m1 = self.mb.create_meshset()
        self.mb.add_entities(m1, self.all_nodes)
        m2 = self.mb.create_meshset()
        self.mb.add_entities(m2, self.all_faces)
        m3 = self.mb.create_meshset()
        self.mb.add_entities(m3, self.all_volumes)
        if text == None:
            text = "output"
        extension = ".vtk"
        text1 = text + "-nodes" + extension
        text2 = text + "-face" + extension
        text3 = text + "-volume" + extension
        self.mb.write_file(text1,[m1])
        self.mb.write_file(text2,[m2])
        self.mb.write_file(text3,[m3])
        print(text, "Arquivos gerados")
#--------------Início dos parâmetros de entrada-------------------
M1= MeshManager('27x27x27.msh')          # Objeto que armazenará as informações da malha
all_volumes=M1.all_volumes

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)
l1=3
l2=9
# Posição aproximada de cada completação
Cent_weels=[[0.5, 0.5, 0.5],[26.5, 26.5, 26.5]]

# Distância, em relação ao poço, até onde se usa malha fina
r0=5

# Distância, em relação ao poço, até onde se usa malha intermediária
r1=7
#--------------fim dos parâmetros de entrada------------------------------------
print("")
print("INICIOU PRÉ PROCESSAMENTO")
tempo0_pre=time.time()
def Min_Max(e):
    verts = M1.mb.get_connectivity(e)     #Vértices de um elemento da malha fina
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
    return([xmin,xmax,ymin,ymax,zmin,zmax])

#--------------Definição das dimensões dos elementos da malha fina--------------
# Esse bloco deve ser alterado para uso de malhas não estruturadas
all_volumes=M1.all_volumes
print("Volumes:",all_volumes)
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
print("definiu dimensões")
# ----- Definição dos volumes que pertencem à malha fina e armazenamento em uma lista----
# finos -> Lista qua armazena os volumes com completação e também aqueles com distância (em relação ao centroide)
#aos volumes com completação menor que "r0"
finos=[]
intermediarios=[]
pocos_meshset=M1.mb.create_meshset()

# Determina se cada um dos elementos está a uma distância inferior a "r0" de alguma completação
# O quadrado serve para pegar os volumes qualquer direção
for e in all_volumes:
    centroid=M1.mtu.get_average_position([e])
    # Cent_wells -> Lista com o centroide de cada completação
    for c in Cent_weels:
        dx=(centroid[0]-c[0])**2
        dy=(centroid[1]-c[1])**2
        dz=(centroid[2]-c[2])**2
        distancia=dx+dy+dz
        if distancia<r0**2:
            finos.append(e)
            if dx<dx0/4+.1 and dy<dy0/4+.1 and dz<dz0/4+.1:
                M1.mb.add_entities(pocos_meshset,[e])
        if distancia<r1**2 and dx<r1**2/2:
            intermediarios.append(e)
M1.mb.tag_set_data(M1.finos_tag, 0,pocos_meshset)

print("definiu volumes na malha fina")

pocos=M1.mb.get_entities_by_handle(pocos_meshset)
volumes_d = [pocos[0]]
volumes_n = [pocos[1]]

finos_meshset = M1.mb.create_meshset()

print("definiu poços")
#-------------------------------------------------------------------------------
#Determinação das dimensões do reservatório e dos volumes das malhas intermediária e grossa
for v in M1.all_nodes:       # M1.all_nodes -> todos os vértices da malha fina
    c=M1.mb.get_coords([v])  # Coordenadas de um nó
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[2]
    if c[2]<zmin: zmin=c[2]

Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin  # Dimensões do reservatório
#-------------------------------------------------------------------------------

# Criação do vetor que define a "grade" que separa os volumes da malha grossa
# Essa grade é absoluta (relativa ao reservatório como um todo)
lx2, ly2, lz2 = [], [], []
# O valor 0.01 é adicionado para corrigir erros de ponto flutuante
for i in range(int(Lx/l2+1.01)):    lx2.append(xmin+i*l2)
for i in range(int(Ly/l2+1.01)):    ly2.append(ymin+i*l2)
for i in range(int(Lz/l2+1.01)):    lz2.append(zmin+i*l2)

#-------------------------------------------------------------------------------
press = 100.0
vazao = 10.0
dirichlet_meshset = M1.mb.create_meshset()
neumann_meshset = M1.mb.create_meshset()

if M1.gravity == False:
    pressao = np.repeat(press, len(volumes_d))

# # colocar gravidade
elif M1.gravity == True:
    pressao = []
    z_elems_d = -1*np.array([M1.mtu.get_average_position([v])[2] for v in volumes_d])
    delta_z = z_elems_d + Lz
    pressao = M1.gama*(delta_z) + press
###############################################
else:
    print("Defina se existe gravidade (True) ou nao (False)")

M1.mb.add_entities(dirichlet_meshset, volumes_d)
M1.mb.add_entities(neumann_meshset, volumes_n)
M1.mb.add_entities(finos_meshset, volumes_n + volumes_d)

#########################################################################################
#jp: modifiquei as tags para sparse
neumann=M1.mb.tag_get_handle("neumann", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
dirichlet=M1.mb.tag_get_handle("dirichlet", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
###############################################################################################

M1.mb.tag_set_data(neumann, volumes_n, np.repeat(1, len(volumes_n)))
M1.mb.tag_set_data(dirichlet, volumes_d, np.repeat(1, len(volumes_d)))

M1.mb.tag_set_data(M1.wells_neumann_tag, 0, neumann_meshset)
M1.mb.tag_set_data(M1.wells_dirichlet_tag, 0, dirichlet_meshset)
M1.mb.tag_set_data(M1.finos_tag, 0, finos_meshset)
M1.mb.tag_set_data(M1.press_value_tag, volumes_d, pressao)
M1.mb.tag_set_data(M1.vazao_value_tag, volumes_n, np.repeat(vazao, len(volumes_n)))

#-------------------------------------------------------------------------------
lxd2=[lx2[0]+l1/2]
if len(lx2)>2:
    for i in np.linspace((lx2[1]+lx2[2])/2,(lx2[-2]+lx2[-3])/2,len(lx2)-3):
        lxd2.append(i)
lxd2.append(lx2[-1]-l1/2)

lyd2=[ly2[0]+l1/2]
if len(ly2)>2:
    for i in np.linspace((ly2[1]+ly2[2])/2,(ly2[-2]+ly2[-3])/2,len(ly2)-3):
        lyd2.append(i)
lyd2.append(ly2[-1]-l1/2)

lzd2=[lz2[0]+l1/2]
if len(lz2)>2:
    for i in np.linspace((lz2[1]+lz2[2])/2,(lz2[-2]+lz2[-3])/2,len(lz2)-3):
        lzd2.append(i)
lzd2.append(lz2[-1]-l1/2)

print("definiu planos do nível 2")

# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(int(l2/l1)):   lx1.append(i*l1)
for i in range(int(l2/l1)):   ly1.append(i*l1)
for i in range(int(l2/l1)):   lz1.append(i*l1)

lxd1=[xmin+dx0/100]
for i in np.linspace(xmin+1.5*l1,xmax-1.5*l1,int((Lx-3*l1)/l1+1.1)):
    lxd1.append(i)
lxd1.append(xmin+Lx-dx0/100)

lyd1=[ymin+dy0/100]
for i in np.linspace(ymin+1.5*l1,ymax-1.5*l1,int((Ly-3*l1)/l1+1.1)):
    lyd1.append(i)
lyd1.append(ymin+Ly-dy0/100)

lzd1=[zmin+dz0/100]
for i in np.linspace(zmin+1.5*l1,zmax-1.5*l1,int((Lz-3*l1)/l1+1.1)):
    lzd1.append(i)
lzd1.append(xmin+Lz-dz0/100)

print("definiu planos do nível 1")
node=M1.all_nodes[0]
coords=M1.mb.get_coords([node])
min_dist_x=coords[0]
min_dist_y=coords[1]
min_dist_z=coords[2]
#-------------------------------------------------------------------------------
'''
#----Correção do posicionamento dos planos que definem a dual
# Evita que algum nó pertença a esses planos e deixe as faces da dual descontínuas
for i in range(len(lxd1)):
    for j in range(len(lyd1)):
        for k in range(len(lzd1)):
            for n in range(len(M1.all_nodes)):
                coord=M1.mb.get_coords([M1.all_nodes[n]])
                dx=lxd1[i]-coord[0]
                dy=lyd1[j]-coord[1]
                dz=lzd1[k]-coord[2]
                if np.abs(dx)<0.0000001:
                    print('Plano x = ',lxd1[i],'corrigido com delta x = ',dx)
                    lxd1[i]-=0.000001
                    i-=1
                if np.abs(dy)<0.0000001:
                    print('Plano y = ',lyd1[j],'corrigido com delta y = ',dy)
                    lyd1[j]-=0.000001
                    j-=1
                if np.abs(dz)<0.0000001:
                    print('Plano z = ',lzd1[k],'corrigido dom delta z = ', dz)
                    lzd1[k]-=0.000001
                    k-=1
#-------------------------------------------------------------------------------
print("corrigiu planos do nível 1")'''
t0=time.time()
# ---- Criação e preenchimento da árvore de meshsets----------------------------
# Esse bloco é executado apenas uma vez em um problema bifásico, sua eficiência
# não é criticamente importante.
L2_meshset=M1.mb.create_meshset()       # root Meshset
D2_meshset=M1.mb.create_meshset()

###########################################################################################
#jp:modifiquei as tags abaixo para sparse
D1_tag=M1.mb.tag_get_handle("d1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
D2_tag=M1.mb.tag_get_handle("d2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################

fine_to_primal1_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL1_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
fine_to_primal2_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL2_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
AV_meshset=M1.mb.create_meshset()

primal_id_tag1 = M1.mb.tag_get_handle("PRIMAL_ID_1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
primal_id_tag2 = M1.mb.tag_get_handle("PRIMAL_ID_2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
nc1=0
nc2 = 0
lc1 = {}
#lc1 é um mapeamento de ids no nivel 1 para o meshset correspondente
#{id:meshset}
for e in all_volumes: M1.mb.add_entities(AV_meshset,[e])
for i in range(len(lx2)-1):
    t1=time.time()
    for j in range(len(ly2)-1):
        for k in range(len(lz2)-1):
            volumes_nv2 = []
            l2_meshset=M1.mb.create_meshset()
            d2_meshset=M1.mb.create_meshset()
            all_volumes=M1.mb.get_entities_by_handle(AV_meshset)
            for elem in all_volumes:
                centroid=M1.mtu.get_average_position([elem])
                if (centroid[0]>lx2[i]) and (centroid[0]<ly2[i]+l2) and (centroid[1]>ly2[j])\
                and (centroid[1]<ly2[j]+l2) and (centroid[2]>lz2[k]) and (centroid[2]<lz2[k]+l2):
                    M1.mb.add_entities(l2_meshset,[elem])
                    M1.mb.remove_entities(AV_meshset,[elem])
                    elem_por_L2=M1.mb.get_entities_by_handle(l2_meshset)

                if i<(len(lxd2)-1) and j<(len(lyd2)-1) and k<(len(lzd2)-1):
                    if (centroid[0]>lxd2[i]-l1/2) and (centroid[0]<lxd2[i+1]+l1/2) and (centroid[1]>lyd2[j]-l1/2)\
                    and (centroid[1]<lyd2[j+1]+l1/2) and (centroid[2]>lzd2[k]-l1/2) and (centroid[2]<lzd2[k+1]+l1/2):

                        M1.mb.add_entities(d2_meshset,[elem])
                        f1a2v3=0
                        if (centroid[0]-lxd2[i])**2<l1**2/4 or (centroid[0]-lxd2[i+1])**2<l1**2/4 :
                            f1a2v3+=1
                        if (centroid[1]-lyd2[j])**2<l1**2/4 or (centroid[1]-lyd2[j+1])**2<l1**2/4:
                            f1a2v3+=1
                        if (centroid[2]-lzd2[k])**2<l1**2/4 or (centroid[2]-lzd2[k+1])**2<l1**2/4:
                            f1a2v3+=1
                        M1.mb.tag_set_data(D2_tag, elem, f1a2v3)
                        M1.mb.tag_set_data(fine_to_primal2_classic_tag, elem, nc2)
            M1.mb.add_child_meshset(L2_meshset,l2_meshset)
            sg=M1.mb.get_entities_by_handle(l2_meshset)
            print(k, len(sg), time.time()-t1)
            t1=time.time()
            d1_meshset=M1.mb.create_meshset()

            M1.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)
            nc2+=1

            for m in range(len(lx1)):
                a=l1*i+m
                for n in range(len(ly1)):
                    b=l1*j+n
                    for o in range(len(lz1)):
                        c=l1*k+o
                        l1_meshset=M1.mb.create_meshset()
                        for e in elem_por_L2:
                            # Refactory here
                            # Verificar se o uso de um vértice reduz o custo
                            centroid=M1.mtu.get_average_position([e])
                            if (centroid[0]>lx2[i]+lx1[m]) and (centroid[0]<lx2[i]+lx1[m]+l1)\
                            and (centroid[1]>ly2[j]+ly1[n]) and (centroid[1]<ly2[j]+ly1[n]+l1)\
                            and (centroid[2]>lz2[k]+lz1[o]) and (centroid[2]<lz2[k]+lz1[o]+l1):
                                M1.mb.add_entities(l1_meshset,[e])
                            if a<(len(lxd1)-1) and b<(len(lyd1)-1) and c<(len(lzd1)-1):
                                if (centroid[0]>lxd1[a]-1.5*dx0) and (centroid[0]<lxd1[a+1]+1.5*dx0)\
                                and (centroid[1]>lyd1[b]-1.5*dy0) and (centroid[1]<lyd1[b+1]+1.5*dy0)\
                                and (centroid[2]>lzd1[c]-1.5*dz0) and (centroid[2]<lzd1[c+1]+1.5*dz0):
                                    M1.mb.add_entities(d1_meshset,[elem])
                                    f1a2v3=0
                                    M_M=Min_Max(e)
                                    if (M_M[0]<lxd1[a] and M_M[1]>lxd1[a]) or (M_M[0]<lxd1[a+1] and M_M[1]>lxd1[a+1]):
                                        f1a2v3+=1
                                    if (M_M[2]<lyd1[b] and M_M[3]>lyd1[b]) or (M_M[2]<lyd1[b+1] and M_M[3]>lyd1[b+1]):
                                        f1a2v3+=1
                                    if (M_M[4]<lzd1[c] and M_M[5]>lzd1[c]) or (M_M[4]<lzd1[c+1] and M_M[5]>lzd1[c+1]):
                                        f1a2v3+=1
                                    M1.mb.tag_set_data(D1_tag, e,f1a2v3)
                                    M1.mb.tag_set_data(fine_to_primal1_classic_tag, e, nc1)


                        M1.mb.tag_set_data(primal_id_tag1, l1_meshset, nc1)
                        nc1+=1
                        M1.mb.add_child_meshset(l2_meshset,l1_meshset)
#-------------------------------------------------------------------------------
all_volumes=M1.all_volumes

vert_meshset=M1.mb.create_meshset()

for e in all_volumes:
    elem_tags = M1.mb.tag_get_tags_on_entity(e)
    d1_tag = int(M1.mb.tag_get_data(elem_tags[2], e))
    if d1_tag==3:
        M1.mb.add_entities(vert_meshset,[e])

print('Criação da árvore: ',time.time()-t0)
print("TEMPO TOTAL DE PRÉ PROCESSAMENTO:",time.time()-tempo0_pre)
print(" ")


t0=time.time()
# --------------Atribuição dos IDs de cada nível em cada volume-----------------
# Esse bloco é executado uma vez a cada iteração em um problema bifásico,
# sua eficiência é criticamente importante.

##########################################################################################
# Tag que armazena o ID do volume no nível 1
# jp: modifiquei as tags abaixo para o tipo sparse
L1_ID_tag=M1.mb.tag_get_handle("l1_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# L1ID_tag=M1.mb.tag_get_handle("l1ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# Tag que armazena o ID do volume no nível 2
L2_ID_tag=M1.mb.tag_get_handle("l2_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# L2ID_tag=M1.mb.tag_get_handle("l2ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# ni = ID do elemento no nível i
L3_ID_tag=M1.mb.tag_get_handle("l3_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################
# ni = ID do elemento no nível i
n1=0
n2=0
aux=0
meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
print("  ")
print("INICIOU SOLUÇÃO ADM")
tempo0_ADM=time.time()
for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1=M1.mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = M1.mb.get_entities_by_handle(m1)
        for elem1 in elem_by_L1:
            if elem1 in finos:
                aux=1
                tem_poço_no_vizinho=True
            if elem1 in intermediarios:
                tem_poço_no_vizinho=True
        if aux==1:
            aux=0
            for elem in elem_by_L1:
                n1+=1
                n2+=1

                M1.mb.tag_set_data(L1_ID_tag, elem, n1)
                M1.mb.tag_set_data(L2_ID_tag, elem, n2)
                M1.mb.tag_set_data(L3_ID_tag, elem, 1)
                elem_tags = M1.mb.tag_get_tags_on_entity(elem)
                elem_Global_ID = M1.mb.tag_get_data(elem_tags[0], elem, flat=True)
                finos.append(elem)

    if tem_poço_no_vizinho:
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            n1+=1
            n2+=1
            t=1
            for elem in elem_by_L1:
                if elem not in finos:
                    M1.mb.tag_set_data(L1_ID_tag, elem, n1)
                    M1.mb.tag_set_data(L2_ID_tag, elem, n2)
                    M1.mb.tag_set_data(L3_ID_tag, elem, 2)
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
                M1.mb.tag_set_data(L2_ID_tag, elem2, n2)
                M1.mb.tag_set_data(L1_ID_tag, elem2, n1)
                M1.mb.tag_set_data(L3_ID_tag, elem2, 3)

# ------------------------------------------------------------------------------
print('Definição da malha ADM: ',time.time()-t0)
t0=time.time()

av=M1.mb.create_meshset()
for v in all_volumes:
    M1.mb.add_entities(av,[v])

# fazendo os ids comecarem de 0 em todos os niveis
tags = [L1_ID_tag, L2_ID_tag]
for tag in tags:
    all_gids = M1.mb.tag_get_data(tag, M1.all_volumes, flat=True)
    minim = min(all_gids)
    all_gids -= minim
    M1.mb.tag_set_data(tag, M1.all_volumes, all_gids)

# volumes da malha grossa primal 1
meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))

# volumes da malha grossa primal 2
meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))

for meshset in meshsets_nv1:
    nc = M1.mb.tag_get_data(primal_id_tag1, meshset, flat=True)[0]
    elems = M1.mb.get_entities_by_handle(meshset)
    M1.mb.tag_set_data(fine_to_primal1_classic_tag, elems, np.repeat(nc, len(elems)))

for meshset in meshsets_nv2:
    nc = M1.mb.tag_get_data(primal_id_tag2, meshset, flat=True)[0]
    elems = M1.mb.get_entities_by_handle(meshset)
    M1.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))

internos_meshset=M1.mb.create_meshset()
faces_meshset=M1.mb.create_meshset()
arestas_meshset=M1.mb.create_meshset()
vertices_meshset=M1.mb.create_meshset()

print("Obtendo OP_ADM_1")
t0=time.time()
for v in M1.all_volumes:
    vafi = M1.mb.tag_get_data(D1_tag, v, flat=True)[0] #Vértice ->3 aresta->2
    if vafi==0:
        M1.mb.add_entities(internos_meshset,[v])
    elif vafi==1:
        M1.mb.add_entities(faces_meshset,[v])
    elif vafi==2:
        M1.mb.add_entities(arestas_meshset,[v])
    elif vafi==3:
        M1.mb.add_entities(vertices_meshset,[v])

internos=M1.mb.get_entities_by_handle(internos_meshset)
faces=M1.mb.get_entities_by_handle(faces_meshset)
arestas=M1.mb.get_entities_by_handle(arestas_meshset)
vertices=M1.mb.get_entities_by_handle(vertices_meshset)
ni=len(internos)
nf=len(faces)
na=len(arestas)
nv=len(vertices)
id=0
for i in internos:
    M1.mb.tag_set_data(M1.ID_reordenado_tag,i,id)
    id+=1
for f in faces:
    M1.mb.tag_set_data(M1.ID_reordenado_tag,f,id)
    id+=1
for a in arestas:
    M1.mb.tag_set_data(M1.ID_reordenado_tag,a,id)
    id+=1
for v in vertices:
    M1.mb.tag_set_data(M1.ID_reordenado_tag,v,id)
    id+=1
T=np.zeros((len(M1.all_volumes),len(M1.all_volumes)))
for f in M1.all_faces:
    adjs = M1.mtu.get_bridge_adjacencies(f, 2, 3)
    if len(adjs)>1:
        Gid_1=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[0]))
        Gid_2=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[1]))
        T[Gid_1][Gid_2]=1
        T[Gid_2][Gid_1]=1
        T[Gid_1][Gid_1]-=1
        T[Gid_2][Gid_2]-=1

Aii=T[0:ni,0:ni]
Aif=T[0:ni,ni:ni+nf]
Aff=T[ni:ni+nf,ni:ni+nf]
for i in range(len(Aff)):
    Aff[i][i] += sum(np.transpose(Aif)[i])
Afe=T[ni:ni+nf,ni+nf:ni+nf+na]
Aee=T[ni+nf:ni+nf+na,ni+nf:ni+nf+na]
Avv=T[ni+nf+na:ni+nf+na+nv,ni+nf+na:ni+nf+na+nv]
for i in range(len(Aee)):
    Aee[i][i] += sum(np.transpose(Afe)[i])
Aev=T[ni+nf:ni+nf+na,ni+nf+na:ni+nf+na+nv]
Ivv=np.identity(nv)
P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)
P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)
P=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P[0:nf,0:nv])),P),axis=0)

AMS_TO_ADM={}
i=0
for v in vertices:
    ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
    AMS_TO_ADM[str(i)] = ID_ADM
    i+=1

OP_ADM=np.zeros((len(M1.all_volumes),n1))
for v in M1.all_volumes:
    nivel=int(M1.mb.tag_get_data(L3_ID_tag,v))
    d1=int(M1.mb.tag_get_data(D1_tag,v))
    ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    if nivel==1 or d1==3:
        OP_ADM[ID_global][ID_ADM]=1
    else:
        for i in range(len(P[ID_global])):
            p=P[ID_global][i]
            if p>0:
                id_ADM=AMS_TO_ADM[str(i)]
                OP_ADM[ID_global][id_ADM]=p
print("took",time.time()-t0)
print("Obtendo OP_ADM_2")
t0=time.time()
OR_ADM=np.zeros((n1,len(M1.all_volumes)),dtype=np.int)
for v in all_volumes:
     elem_Global_ID = int(M1.mb.tag_get_data(M1.ID_reordenado_tag, v, flat=True))
     elem_ID1 = int(M1.mb.tag_get_data(L1_ID_tag, v, flat=True))
     OR_ADM[elem_ID1][elem_Global_ID]=1


OR_AMS=np.zeros((nv,len(M1.all_volumes)),dtype=np.int)
for v in all_volumes:
     elem_Global_ID = int(M1.mb.tag_get_data(M1.ID_reordenado_tag, v))
     AMS_ID = int(M1.mb.tag_get_data(fine_to_primal1_classic_tag, v))
     OR_AMS[AMS_ID][elem_Global_ID]=1

COL_TO_AMS={}
i=0
for v in vertices:
    ID_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag,v)
    COL_TO_AMS[str(i)] = ID_AMS
    i+=1

OP_AMS=np.zeros((len(M1.all_volumes),nv))
for v in M1.all_volumes:
    elem_Global_ID = int(M1.mb.tag_get_data(M1.ID_reordenado_tag, v))
    for i in range(len(P[elem_Global_ID])):
        p=P[elem_Global_ID][i]
        if p>0:
            ID_AMS=COL_TO_AMS[str(i)]
            OP_AMS[elem_Global_ID][ID_AMS]=p

#REPETIDO -------------------------------------
T=np.zeros((len(M1.all_volumes),len(M1.all_volumes)))
for f in M1.all_faces:
    adjs = M1.mtu.get_bridge_adjacencies(f, 2, 3)
    if len(adjs)>1:
        Gid_1=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[0]))
        Gid_2=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[1]))
        T[Gid_1][Gid_2]=1
        T[Gid_2][Gid_1]=1
        T[Gid_1][Gid_1]-=1
        T[Gid_2][Gid_2]-=1
#----------------------------------------------------
T_AMS=np.dot(np.dot(OR_AMS,T),OP_AMS)
T_ADM=np.dot(np.dot(OR_ADM,T),OP_ADM)

int_meshset=M1.mb.create_meshset()
fac_meshset=M1.mb.create_meshset()
are_meshset=M1.mb.create_meshset()
ver_meshset=M1.mb.create_meshset()
for v in vertices:
    vafi = M1.mb.tag_get_data(D2_tag, v)[0] #Vértice ->3 aresta->2
    if vafi==0:
        M1.mb.add_entities(int_meshset,[v])
    elif vafi==1:
        M1.mb.add_entities(fac_meshset,[v])
    elif vafi==2:
        M1.mb.add_entities(are_meshset,[v])
    elif vafi==3:
        M1.mb.add_entities(ver_meshset,[v])
G=np.zeros((nv,nv))
inte=M1.mb.get_entities_by_handle(int_meshset)
fac=M1.mb.get_entities_by_handle(fac_meshset)
are=M1.mb.get_entities_by_handle(are_meshset)
ver=M1.mb.get_entities_by_handle(ver_meshset)
nint=len(inte)
nfac=len(fac)
nare=len(are)
nver=len(ver)
for i in range(nint):
    v=inte[i]
    ID_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag,v)
    G[i][ID_AMS]=1
i=0
for i in range(nfac):
    v=fac[i]
    ID_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag,v)
    G[nint+i][ID_AMS]=1
i=0
for i in range(nare):
    v=are[i]
    ID_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag,v)
    G[nint+nfac+i][ID_AMS]=1
i=0

for i in range(nver):
    v=ver[i]
    ID_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag,v)
    G[nint+nfac+nare+i][ID_AMS]=1

W=np.dot(G,T_AMS)
W_AMS=np.dot(W,np.transpose(G))

MPFA_NO_NIVEL_2=True
#-------------------------------------------------------------------------------
ni=nint
nf=nfac
na=nare
nv=nver

Aii=W_AMS[0:ni,0:ni]
Aif=W_AMS[0:ni,ni:ni+nf]
Aie=W_AMS[0:ni,ni+nf:ni+nf+na]
Aiv=W_AMS[0:ni,ni+nf+na:ni+nf+na+nv]
for i in range(len(Aii)):
    if MPFA_NO_NIVEL_2 ==False:
        Aii[i][i] += sum(Aie[i])+sum((Aiv)[i])

Afi=W_AMS[ni:ni+nf,0:ni]
Aff=W_AMS[ni:ni+nf,ni:ni+nf]
Afe=W_AMS[ni:ni+nf,ni+nf:ni+nf+na]
Afv=W_AMS[ni:ni+nf,ni+nf+na:ni+nf+na+nv]
for i in range(len(Aff)):
    Aff[i][i] += sum(Afi[i])
    if MPFA_NO_NIVEL_2==False:
        Aff[i][i] +=sum(Afv[i])

Aei=W_AMS[ni+nf:ni+nf+na,0:ni]
Aef=W_AMS[ni+nf:ni+nf+na,ni:ni+nf]
Aee=W_AMS[ni+nf:ni+nf+na,ni+nf:ni+nf+na]
Aev=W_AMS[ni+nf:ni+nf+na,ni+nf+na:ni+nf+na+nv]
for i in range(len(Aee)):
    Aee[i][i] += sum(Aei[i])+sum(Aef[i])

Avv=W_AMS[ni+nf+na:ni+nf+na+nv,ni+nf+na:ni+nf+na+nv]
Ivv=np.identity(nv)
P2=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

if MPFA_NO_NIVEL_2:
    invAff=np.linalg.inv(Aff)
    M2=-np.dot(invAff,np.dot(Afe,P2[0:na,0:nv]))-np.dot(invAff,Afv)
    P2=np.concatenate((M2,P2), axis=0)
else:
    P2=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P2[0:na,0:nv])),P2), axis=0)
if MPFA_NO_NIVEL_2:
    M3=np.dot(np.linalg.inv(Aii),(-np.dot(Aif,M2)+np.dot(Aie,np.dot(np.linalg.inv(Aee),Aev))-Aiv))
    P2=np.concatenate((M3,P2), axis=0)
else:
    P2=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P2[0:nf,0:nv])),P2),axis=0)

OP_AMS_2=np.dot(np.transpose(G),P2)

COL_TO_ADM_2={}
# ver é o meshset dos vértices da malha dual grossa
for v in ver:
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal2_classic_tag,v))
    ID_ADM=int(M1.mb.tag_get_data(L2_ID_tag,v))
    COL_TO_ADM_2[str(ID_AMS)] = ID_ADM

#---Vértices é o meshset dos véttices da malha dual do nivel intermediário------

OP_ADM_2=np.zeros((len(T_ADM),n2))
print("took",time.time()-t0)
print("Resolvendo sistema ADM_2")
t0=time.time()
My_IDs_2=[]
for v in M1.all_volumes:
    ID_global=int(M1.mb.tag_get_data(L1_ID_tag,v))
    if ID_global not in My_IDs_2:
        My_IDs_2.append(ID_global)
        ID_ADM=int(M1.mb.tag_get_data(L2_ID_tag,v))
        nivel=M1.mb.tag_get_data(L3_ID_tag,v)
        d1=M1.mb.tag_get_data(D2_tag,v)
        ID_AMS = int(M1.mb.tag_get_data(fine_to_primal2_classic_tag, v))
        # nivel<3 refere-se aos volumes na malha fina (nivel=1) e intermédiária (nivel=2)
        # d1=3 refere-se aos volumes que são vértice na malha dual de grossa
        if nivel<3:
            OP_ADM_2[ID_global][ID_ADM]=1
        else:
            for i in range(len(P2[ID_AMS])):
                p=P2[ID_AMS][i]
                if p>0:
                    id_ADM=COL_TO_ADM_2[str(i)]
                    OP_ADM_2[ID_global][id_ADM]=p

OR_ADM_2=np.zeros((n2,len(T_ADM)),dtype=np.int)
for v in M1.all_volumes:
    elem_ID2 = int(M1.mb.tag_get_data(L2_ID_tag, v, flat=True))
    elem_Global_ID = int(M1.mb.tag_get_data(L1_ID_tag, v, flat=True))
    OR_ADM_2[elem_ID2][elem_Global_ID]=1

T_ADM_2=np.dot(np.dot(OR_ADM_2,T_ADM),OP_ADM_2)

#-------------------------------------------------------------------------------
# Insere condições de dirichlet nas matrizes
for i in range(len(volumes_d)):
    v=volumes_d[i]
    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
    ID_ADM_2=int(M1.mb.tag_get_data(L2_ID_tag,v))
    for j in range(len(T[i])):
        T[ID_global][j]=0
        if j==ID_global:
            T[ID_global][j]=1
    for j in range(len(T_ADM[i])):
        T_ADM[ID_ADM][j]=0
        if j==ID_ADM:
            T_ADM[ID_ADM][j]=1
    for j in range(len(T_ADM_2[i])):
        T_ADM_2[ID_ADM_2][j]=0
        if j==ID_ADM_2:
            T_ADM_2[ID_ADM_2][j]=1

# Gera a matriz dos coeficientes
b=np.zeros((len(M1.all_volumes),1))
for d in volumes_d:
    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,d))
    b[ID_global]=press
for n in volumes_n:
    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,n))
    b[ID_global]=vazao

b_ADM=np.dot(OR_ADM,b)

b_ADM_2=np.dot(OR_ADM_2,b_ADM)

t0=time.time()
SOL_ADM_2=np.linalg.solve(T_ADM_2,b_ADM_2)
print("resolveu ADM_2: ",time.time()-t0)
print("Prolongando sol ADM")
t0=time.time()
SOL_ADM_2=np.dot(OP_ADM_2,SOL_ADM_2)
SOL_ADM_fina=np.dot(OP_ADM,SOL_ADM_2)
print("Prolongou sol ADM: ",time.time()-t0)
print("TEMPO TOTAL PARA SOLUÇÃO ADM:", time.time()-tempo0_ADM)
print("")
'''
SOL_TPFA = np.load('SOL_TPFA.npy')
'''
print("resolvendo TPFA")
t0=time.time()
SOL_TPFA=np.linalg.solve(T,b)
print("resolveu TPFA: ",time.time()-t0)
np.save('SOL_TPFA.npy', SOL_TPFA)

erro=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)):
    erro[i]=(SOL_TPFA[i]-SOL_ADM_fina[i])/SOL_TPFA[i]

ERRO_tag=M1.mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_TPFA_tag=M1.mb.tag_get_handle("Pressão TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_ADM_tag=M1.mb.tag_get_handle("Pressão ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

for v in M1.all_volumes:
    gid=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    M1.mb.tag_set_data(ERRO_tag,v,erro[gid])
    M1.mb.tag_set_data(Sol_TPFA_tag,v,SOL_TPFA[gid])
    M1.mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])

M1.mb.write_file('teste_3D_unstructured_18.vtk',[av])
print('New file created')
import pdb; pdb.set_trace()
