import numpy as np

inds_adm = np.load('inds_op_adm.npy')
inds_adm[3][1] += 1


op_adm = np.zeros((inds_adm[3][0], inds_adm[3][1]), dtype=np.float64)
op_adm[inds_adm[0], inds_adm[1]] = inds_adm[2]

det = np.linalg.det(op_adm)
print(det)
import pdb; pdb.set_trace()
