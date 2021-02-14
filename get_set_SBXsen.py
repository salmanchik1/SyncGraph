import numpy as np


def getSBXsen(SBX, component, sen):
    if sen in SBX['field'][0, :]:
        Nsen = np.where(SBX['field'][0, :] == sen)[0][0]
        return SBX[component][Nsen, :]
    else:
        LenOfSignal = SBX['SZ'].shape[1]
        return np.full(LenOfSignal, np.nan)


def setSBXsen(SBX, component, sen, data):
    if sen in SBX['field'][0, :]:
        Nsen = np.where(SBX['field'][0, :] == sen)[0][0]
        SBX[component][Nsen, :] = data
