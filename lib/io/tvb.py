import zipfile
import numpy as np


def read_roi_cntrs(cntrs_zipfile):
    roi_cntrs = []
    roi_lbls = []
    with zipfile.ZipFile(cntrs_zipfile) as zf:
        with zf.open('centres.txt') as t:
            for line in t:
                roi_cntrs.append(line.decode('utf-8').strip().split(' ')[1:])
                roi_lbls.append(line.decode('utf-8').strip().split(' ')[0])
    roi_cntrs = np.array(roi_cntrs, dtype=float)
    return roi_cntrs, roi_lbls


def read_connectome(sc_path, normalize=True, zero_diag=True):
    with zipfile.ZipFile(sc_path) as conn_zip:
        with conn_zip.open('weights.txt') as wts_fd:
            sc = np.loadtxt(wts_fd)
    if(zero_diag):
        sc[np.diag_indices_from(sc)] = 0
    if(normalize):
        sc = sc / sc.max()
    return sc
