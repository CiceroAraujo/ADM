import numpy as np
import time
import os
from others_utils import OtherUtils as oth
from prolongation import ProlongationTPFA3D as prol3d

# os.chdir('/home/joao/Documentos/ADM/ADM/jp_adm/output')

class testing:

    def test_op_adm(self):
        lim = 1e-10

        inds_adm = self.load_array('inds_op_adm.npy')

        self.op_adm = np.zeros((inds_adm[3][0], inds_adm[3][1]), dtype=np.float64)
        self.op_adm[inds_adm[0], inds_adm[1]] = inds_adm[2]

        I1 = np.sum(self.op_adm, axis=1)

        indices = np.where(I1 < lim)[0]

        # print(indices)
        # print(len(indices))
        # print(I1[indices])
        # print('\n')

        indices = np.where(I1 > 1+lim)[0]

        # print(indices)
        # print(len(indices))
        # print(I1[indices])

    def test_tc(self):
        inds_tc_faces = self.load_array('inds_tc1_faces.npy')
        inds_tc = self.load_array('inds_tc1.npy')
        elems_wirebasket_nv1 = self.load_array('elems_wirebasket_nv1.npy')
        indsG_nv1 = self.load_array('inds_G_nv1.npy')
        indsGT_nv1 = self.load_array('inds_GT_nv1.npy')
        inds_tc_wirebasket = self.load_array('inds_tc_wirebasket.npy')
        vss = elems_wirebasket_nv1[0]
        ess = elems_wirebasket_nv1[1]
        fss = elems_wirebasket_nv1[2]
        iss = elems_wirebasket_nv1[3]

        nni = len(iss)
        nnf = nni + len(fss)
        nne = nnf + len(ess)
        nnv = nne + len(vss)
        wirebasket_numbers = [len(iss), len(fss), len(ess), len(vss)]

        sz = inds_tc[3]

        tc = np.zeros((sz[0], sz[1]), dtype=np.float64)
        G = tc.copy()
        GT = tc.copy()
        tc_wire = tc.copy()
        tc_faces = tc.copy()

        tc[inds_tc[0], inds_tc[1]] = inds_tc[2]
        G[indsG_nv1[0], indsG_nv1[1]] = indsG_nv1[2]
        GT[indsGT_nv1[0], indsGT_nv1[1]] = indsGT_nv1[2]
        tc_wire[inds_tc_wirebasket[0], inds_tc_wirebasket[1]] = inds_tc_wirebasket[2]
        tc_wire2 = np.dot(GT, tc)
        tc_wire2 = np.dot(tc_wire2, G)
        tc_faces[inds_tc_faces[0], inds_tc_faces[1]] = inds_tc_faces[2]

        print(np.allclose(tc_wire, tc_wire2))
        import pdb; pdb.set_trace()

        inds_tc_wire2 = np.nonzero(tc_wire2)
        inds_tc_wire2 = [inds_tc_wire2[0], inds_tc_wire2[1], tc_wire2[inds_tc_wire2[0], inds_tc_wire2[1]], sz]


        import pdb; pdb.set_trace()

        inds_tc_tpfa_wirebasket = oth.get_tc_tpfa(inds_tc_wire2, wirebasket_numbers)
        tc_tpfa_wirebasket = np.zeros(tuple(sz))
        tc_tpfa_wirebasket[inds_tc_tpfa_wirebasket[0], inds_tc_tpfa_wirebasket[1]] = inds_tc_tpfa_wirebasket[2]
        tc3 = np.dot(G, tc_tpfa_wirebasket)
        tc3 = np.dot(tc3, GT)

        ff = tc3[nnf:nne, nnf:nne]
        print(np.linalg.det(ff))
        print(np.linalg.cond(ff))
        tt = np.linalg.inv(ff)
        print(np.dot(tt, ff))
        import pdb; pdb.set_trace()
        print(np.allclose(tc3, tc_faces))

        lim = 1e-9

        # teste para verificar igualdade de linhas ou colunas ou linhas ou colunas nulas
        # for i in range(nc):
        #     line1 = tc[i]
        #     col1 = tc[:,i]
        #
        #     if abs(line1.sum()) > lim:
        #         print('erro modulo da soma maior que zero')
        #         print(i)
        #         print(line1.sum())
        #         import pdb; pdb.set_trace()
        #
        #     if np.allclose(np.array(line1[i]), np.array([0.0])):
        #         print('erro diagonal igual a zero')
        #         print(i)
        #         print(line[i])
        #         import pdb; pdb.set_trace()
        #
        #     if np.allclose(line1, zeros):
        #         print('erro linha toda nula')
        #         print(line1)
        #         print(i)
        #         print('\n')
        #         import pdb; pdb.set_trace()
        #     if np.allclose(col1, zeros):
        #         print('erro coluna toda nula')
        #         print(line1)
        #         print(i)
        #         print('\n')
        #         import pdb; pdb.set_trace()
        #
        #
        #     for j in range(i, nc):
        #         if j == i:
        #             continue
        #         line2 = tc[j]
        #         col2 = tc[:,j]
        #         if np.allclose(line1, line2):
        #             print('erro linhas iguais')
        #             print(line1)
        #             print(i)
        #             print(line2)
        #             print(j)
        #             print('\n')
        #             import pdb; pdb.set_trace()
        #         if np.allclose(col1, col2):
        #             print('erro colunas iguais')
        #             print(col1)
        #             print(i)
        #             print(col2)
        #             print(j)
        #             print('\n')
        #             import pdb; pdb.set_trace()

    def test_tc_mod(self):
        inds_tc = self.load_array('inds_tc_mod.npy')
        sz = inds_tc[3]
        elems_wirebasket_nv1 = self.load_array('elems_wirebasket_nv1.npy')
        vss = elems_wirebasket_nv1[0]
        ess = elems_wirebasket_nv1[1]
        fss = elems_wirebasket_nv1[2]
        iss = elems_wirebasket_nv1[3]

        nni = len(iss)
        nnf = nni + len(fss)
        nne = nnf + len(ess)
        nnv = nne + len(vss)

        l = [nni, nnf, nne, nnv]

        tc = np.zeros((sz[0], sz[1]), dtype=np.float64)
        tc[inds_tc[0], inds_tc[1]] = inds_tc[2]

        # for i in range(len(l)):
        #     nn = l[i]
        #     if i == 0:
        #         sl = tc[:nn, :nn]
        #     elif i == len(l)-1:
        #         continue
        #     else:
        #         ant = l[i-1]
        #         sl = tc[ant:nn, ant:nn]
        #
        #     print(sl)
        #     det = np.linalg.det(sl)
        #     ff = np.linalg.inv(sl)
        #     cond = np.linalg.cond(sl)
        #     print(det)
        #     print(cond)
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        for i in range(nnf, nne):
            inds = np.nonzero(tc[i])
            print(inds)
            print(tc[i][inds])
            print(sum(tc[i][inds]))
            print('\n')
            import pdb; pdb.set_trace()



            # lim = 1e-10
            # det = np.linalg.det(sl)
            # els = np.diag(sl)
            # print(np.prod(els))
            # print(det)
            # # print(els)
            # import pdb; pdb.set_trace()
            # for k in range(sl.shape[0]):
            #     tt = np.where((np.absolute(sl[k])) > lim)[0]
            #     if len(tt) < 1:
            #         print('linha nula')
            #         print(sl[k, tt])
            #         import pdb; pdb.set_trace()
            #     tt = np.where(np.absolute(sl[:,k]) > lim)[0]
            #     if len(tt) < 1:
            #         print('coluna nula')
            #         print(sl[tt, k])
            #         import pdb; pdb.set_trace()
            # print('\n')
            # import pdb; pdb.set_trace()

    def test_qc(self):
        self.qc = self.load_array('qc.npy')

    def test_or_adm(self):
        inds_or_adm = self.load_array('inds_or_adm.npy')
        sz = inds_or_adm[3]

        self.or_adm = np.zeros((sz[0], sz[1]))

        self.or_adm[inds_or_adm[0], inds_or_adm[1]] = inds_or_adm[2]

        # cont = 0
        # for i in or_adm:
        #     ind = np.nonzero(i)[0]
        #     print(ind)
        #     print(i[ind])
        #     print(cont)
        #     print('\n')
        #     import pdb; pdb.set_trace()
        #     cont+=1

    def test_op_classic(self):
        inds_op_classic = self.load_array('inds_op1.npy')
        sz = inds_op_classic[3]

        op = np.zeros((sz[0], sz[1]), dtype=np.float64)

        pass

    def test_erro(self):
        self.pf = self.load_array('pf.npy')
        self.pms = self.load_array('pms.npy')

        self.erro = np.absolute(self.pf - self.pms)/self.pf

    def test_pc(self):
        self.pc = self.load_array('pc.npy')

    def test_multiescala(self):

        self.test_or_adm()
        self.test_op_adm()
        self.test_tf()
        self.test_qf()
        self.test_qc()
        self.test_pc()


        lim = 1e-10



        self.tc2 = np.dot(self.or_adm, self.tf)
        self.tc2 = np.dot(self.tc2, self.op_adm)
        self.qc2 = np.dot(self.or_adm, self.qf)
        self.pc2 = np.linalg.solve(self.tc2, self.qc2)
        self.pms2 = np.dot(self.op_adm, self.pc2)

    def test_tf(self):

        lim = 1e-10

        inds_tf = self.load_array('inds_transfine.npy')

        self.tf = np.zeros((inds_tf[3][0], inds_tf[3][1]), dtype=np.float64)
        self.tf[inds_tf[0], inds_tf[1]] = inds_tf[2]

    def test_qf(self):
        self.qf = self.load_array('b.npy')

    def test_tc_wire(self):
        inds_tc_wirebasket = self.load_array('inds_tc_wirebasket.npy')
        elems_wirebasket_nv1 = self.load_array('elems_wirebasket_nv1.npy')
        indsG_nv1 = self.load_array('inds_G_nv1.npy')
        indsGT_nv1 = self.load_array('inds_GT_nv1.npy')
        inds_tc1 = self.load_array('inds_tc1.npy')
        vss = elems_wirebasket_nv1[0]
        ess = elems_wirebasket_nv1[1]
        fss = elems_wirebasket_nv1[2]
        iss = elems_wirebasket_nv1[3]

        nni = len(iss)
        nnf = nni + len(ess)
        nne = nnf + len(fss)
        nnv = nne + len(vss)

        l = [nni, nnf, nne, nnv]

        sz = inds_tc1[3]

        tc1 = np.zeros((sz[0], sz[1]), dtype=np.float64)
        G = tc1.copy()
        GT = tc1.copy()
        tc_wire = tc1.copy()

        tc1[inds_tc1[0], inds_tc1[1]] = inds_tc1[2]
        G[indsG_nv1[0], indsG_nv1[1]] = indsG_nv1[2]
        GT[indsGT_nv1[0], indsGT_nv1[1]] = indsGT_nv1[2]
        tc_wire[inds_tc_wirebasket[0], inds_tc_wirebasket[1]] = inds_tc_wirebasket[2]

        tc_wire2 = np.dot(GT, tc1)
        tc_wire2 = np.dot(tc_wire2, G)
        print(np.allclose(tc_wire2, tc_wire))
        import pdb; pdb.set_trace()
        nc = inds_tc[3][0]
        zeros = np.zeros(nc)

        for i in range(len(l)):
            nn = l[i]
            if i == 0:
                sl = tc[:nn, :nn]
            elif i == len(l)-1:
                continue
            else:
                ant = l[i-1]
                sl = tc[ant:nn, ant:nn]

            # print(sl)
            # det = np.linalg.det(sl)
            # ff = np.linalg.inv(sl)
            # cond = np.linalg.cond(sl)
            # print(det)
            # print(cond)

            lim = 1e-10
            det = np.linalg.det(sl)
            els = np.diag(sl)
            print(np.prod(els))
            print(det)
            # print(els)
            import pdb; pdb.set_trace()
            for k in range(sl.shape[0]):
                tt = np.where((np.absolute(sl[k])) > lim)[0]
                if len(tt) < 1:
                    print('linha nula')
                    print(sl[k, tt])
                    import pdb; pdb.set_trace()
                tt = np.where(np.absolute(sl[:,k]) > lim)[0]
                if len(tt) < 1:
                    print('coluna nula')
                    print(sl[tt, k])
                    import pdb; pdb.set_trace()
            print('\n')
            import pdb; pdb.set_trace()

    def test_op_nv2(self):
        inds_tc1 = self.load_array('inds_tc1.npy')
        inds_tc = self.load_array('inds_tc_mod.npy')
        sz = inds_tc[3]
        elems_wirebasket_nv1 = self.load_array('elems_wirebasket_nv1.npy')
        vss = elems_wirebasket_nv1[0]
        ess = elems_wirebasket_nv1[1]
        fss = elems_wirebasket_nv1[2]
        iss = elems_wirebasket_nv1[3]

        nni = len(iss)
        nnf = nni + len(fss)
        nne = nnf + len(ess)
        nnv = nne + len(vss)

        l = [nni, nnf, nne, nnv]

        tc = np.zeros((sz[0], sz[1]), dtype=np.float64)
        tc[inds_tc[0], inds_tc[1]] = inds_tc[2]

        op_nv2 = np.zeros((nnv, len(vss)))

        lim = 1e-12
        op_nv2[nne:nnv] = np.identity(len(vss))

        inv = np.linalg.inv(tc[nnf:nne, nnf:nne])
        inv = -inv.dot(tc[nnf:nne, nne:nnv])
        k = 1/np.sum(inv, axis=1)
        inv = np.multiply(inv.T, k).T
        op_nv2[nnf:nne] = inv

        inv = np.linalg.inv(tc[nni:nnf, nni:nnf])
        inv = -inv.dot(tc[nni:nnf, nnf:nne])
        inv = inv.dot(op_nv2[nnf:nne])
        # tot = np.sum(inv, axis=1)
        k = 1/np.sum(inv, axis=1)
        inv = np.multiply(inv.T, k).T
        op_nv2[nni:nnf] = inv

        inv = np.linalg.inv(tc[0:nni, 0:nni])
        inv = -inv.dot(tc[0:nni, nni:nnf])
        inv = inv.dot(op_nv2[nni:nnf])
        k = 1/np.sum(inv, axis=1)
        inv = np.multiply(inv.T, k).T
        op_nv2[0:nni] = inv
        self.write_array('op_nv2_numpy_wire', op_nv2)





    @staticmethod
    def load_array(name):
        os.chdir('/pytest/output')
        s = np.load(name)
        os.chdir('/pytest')
        return s

    @staticmethod
    def write_array(name, arr):
        os.chdir('/pytest/output')
        np.save(name, arr)
        os.chdir('/pytest')


test1 = testing()
# test1.test_multiescala()
# test1.test_erro()
# test1.test_op_nv2()
test1.test_tc()
