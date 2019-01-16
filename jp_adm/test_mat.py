import numpy as np
import time
import os

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
        inds_tc = self.load_array('inds_tc1_faces.npy')
        elems_wirebasket_nv1 = self.load_array('elems_wirebasket_nv1.npy')
        indsG_nv1 = self.load_array

        sz = inds_tc[3]

        self.tc = np.zeros((sz[0], sz[1]), dtype=np.float64)

        self.tc[inds_tc[0], inds_tc[1]] = inds_tc[2]

        for i in tc:

            ind = np.nonzero(i)[0]
            print(ind)
            print(i[ind])

            print(sum(i))
            print('\n')
            import pdb; pdb.set_trace()

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

    @staticmethod
    def load_array(name):
        os.chdir('/pytest/output')
        s = np.load(name)
        os.chdir('/pytest')
        return s


test1 = testing()
# test1.test_multiescala()
# test1.test_erro()
test1.test_tc()
