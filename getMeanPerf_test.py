# -*- encoding: utf-8 -*-

import os

import numpy as np
import h5py
from filter_f import filter_f
from scipy.signal import medfilt2d
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
from itertools import cycle
from get_set_SBXsen import getSBXsen, setSBXsen


def load_h5_files(file_paths):
    SBXi = dict()
    for nSBX, file_path in enumerate(file_paths):
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            print(keys)
            SBXf = f.get(keys[0])
            SBXi[nSBX] = dict()
            for key in SBXf:
                SBXi[nSBX][key] = np.array(SBXf[key])
    return SBXi


def calc_vars(SBXi, Fpass1, Fpass2, L, tstrt, dT):
    key = list(SBXi.keys())[0]
    LenOfSignal = SBXi[key]['SZ'].shape[1]

    # создаем множество set() имен сенсоров NamesOfSen из всех входных данных
    NamesOfSen = set()
    for key in SBXi:
        NamesOfSen.update(SBXi[key]['field'][0, :])

    fd = SBXi[key]['fd'][0][0]  # частота дискретизации сигнала
    b = filter_f(fd, Fpass1, Fpass2)
    tstrt = tstrt/fd
    time0 = np.arange(L, dtype=int) + tstrt*fd
    nSBX = len(SBXi)
    T1 = np.zeros((nSBX, L), dtype=int)
    for i in range(nSBX):
        T1[i, :] = dT[i]+time0
    T1[T1 >= LenOfSignal] -= LenOfSignal
    k = np.ones(nSBX)*1000000*1.91037945231066*(10**-8)/fd
    acor = 1  # неизвестная константа?
    NumCh = len(NamesOfSen)
    return LenOfSignal, fd, acor, b, T1, k, NumCh, nSBX, NamesOfSen, time0


def debugging(SBXi, ksen, extraFUP, LenOfSignal, df, fd, acor, b):
    SBX_plot = dict()
    for key in SBXi:
        SBX_plot[key] = dict()
        for XYZ in ['Z', 'X', 'Y']:
            signal = getSBXsen(SBXi[key], 'S'+XYZ, ksen)
            if extraFUP:
                n_fup = int(LenOfSignal*df/fd) if int(LenOfSignal*df/fd) else int(LenOfSignal*df/fd) - 1
                sp = np.fft.fft(signal)
                norsp = abs(sp)
                # medsp = medfilt(norsp, n_fup)
                medsp = medfilt2d(norsp.reshape(1, -1), (1, n_fup))[0]  # на порядок быстрее medfilt
                signal = acor * (np.fft.ifft((sp / norsp) * medsp)).real
            # SBX_plot[key]['F' + XYZ] = lfilter(b, 1, signal)[:, int((len(b) - 1)/2):]
            SBX_plot[key]['F' + XYZ] = filtfilt(b, 1, signal)
    return SBX_plot


def build_plots(SBXi, SBX_plot, T1, k, file_paths):
    # for key in SBXi:
    #     for FZXY in ['FZ', 'FX', 'FY']:
    #         fig, ax = plt.subplots()
    #         ax.plot(SBX_plot[key][FZXY][T1[key, :]] * k[key], label=file_paths[key])
    #         # maxindex = np.argmax(SBX_plot[key][FZXY][T1[key, :]])
    #         # plt.axvline(x=maxindex, label='MAX line at {}'.format(maxindex), c='r')
    #         plt.title(FZXY)
    #         ax.grid()
    #         ax.legend(loc='upper right', ncol=2)
    #         plt.show()

    for FZXY in ['FZ', 'FX', 'FY']:
        fig, ax = plt.subplots()
        for key in SBXi:
            ax.plot(SBX_plot[key][FZXY][T1[key, :]] * k[key], label=file_paths[key], lw=0.8)
        plt.title(FZXY)
        ax.grid()
        ax.legend(loc='upper right', ncol=2)
        plt.show()


def unloading(SBXi, extraFUP, LenOfSignal, df, fd, acor, NumCh, L, nSBX, b, NamesOfSen, Lstd, T1, k, lev, dT, time0):
    for key in SBXi:
        for XYZ in ['Z', 'X', 'Y']:
            if extraFUP:
                n_fup = int(LenOfSignal*df/fd) if int(LenOfSignal*df/fd) else int(LenOfSignal*df/fd) - 1
                for ksen in SBXi[key]['field'][0, :]:
                    print(ksen)
                    sp = np.fft.fft(getSBXsen(SBXi[key], 'S' + XYZ, ksen))
                    norsp = abs(sp)
                    # medsp = medfilt(norsp, n_fup)
                    medsp = medfilt2d(norsp.reshape(1, -1), (1, n_fup))[0]  # на порядок быстрее medfilt
                    setSBXsen(SBXi[key], 'S' + XYZ, ksen, acor * (np.fft.ifft((sp / norsp) * medsp)).real)
            # SBXi[key]['F'+XYZ] = lfilter(b, 1, SBXi[key]['S'+XYZ])[:, int((len(b) - 1)/2):]
            SBXi[key]['F' + XYZ] = filtfilt(b, 1, SBXi[key]['S' + XYZ])
    # создаем усредненный SBX
    SBXm = dict()
    SBXm['SZ'] = np.full((NumCh, L), np.nan)
    SBXm['SX'] = np.full((NumCh, L), np.nan)
    SBXm['SY'] = np.full((NumCh, L), np.nan)
    SBXm['fd'] = fd
    # заполняем field для него из всех входных SBX( field может отличаться, координаты для одинаковых сенсоров тоже)
    SBXm['field'] = np.zeros((4, len(NamesOfSen)))
    SBXm['field'][0, :] = np.array(list(NamesOfSen))
    keys = cycle(list(SBXi.keys()))
    key = next(keys)
    for i in range(len(NamesOfSen)):
        while SBXm['field'][0, i] not in SBXi[key]['field'][0, :]:
            key = next(keys)
        index = np.where(SBXi[key]['field'][0, :] == SBXm['field'][0, i])[0][0]
        SBXm['field'][:, i] = SBXi[key]['field'][:, index]

    inds = np.empty((NumCh * 3, nSBX), dtype=int)
    indsC = np.empty(NumCh * 3, dtype=int)

    for ch in NamesOfSen:
        jj = 0
        # fig, axs = plt.subplots(3, 1, constrained_layout=True)
        # fig.suptitle('Channel number: ' + str(int(SBXi[0]['field'][0, ch])), fontsize=16)

        for ZXY in ['Z', 'X', 'Y']:
            Ss = np.full((nSBX, L), np.nan)    # np.full((2, 3), np.nan)
            Sstd = np.full((nSBX, Lstd), np.nan)
            for i in range(nSBX):
                Ss[i, :] = getSBXsen(SBXi[i], 'F'+ZXY, ch)[T1[i, :]]*k[i]
                Sstd[i, :] = getSBXsen(SBXi[i], 'F' + ZXY, ch)[T1[i, 0:Lstd]] * k[i]
            std = np.std(Sstd, axis=1)
            I = std.argsort(axis=0)
            std.sort(axis=0)
            I = np.concatenate((I[std != 0], I[std == 0]))
            inds[int((ch-1)*3+jj), :] = I
            I = I[std < lev]
            indsC[int((ch-1)*3+jj)] = len(I)

            # plot ?
            # axs[jj].plot(np.mean(Ss[I, :], axis=0)*len(I))
            # axs[jj].set_title(ZXY)
            # axs[jj].axis([0, L, -1, 1])

            setSBXsen(SBXm, 'S'+ZXY, ch, np.mean(Ss[I, :], axis=0))
            jj += 1

        # plt.show()

    dT = dT + time0[0]
    SBXm['SZ'][np.isnan(SBXm['SZ'])] = 0
    SBXm['SX'][np.isnan(SBXm['SX'])] = 0
    SBXm['SY'][np.isnan(SBXm['SY'])] = 0

    for key in SBXi:
        for ZXY in ['Z', 'X', 'Y']:
            SBXi[key]['S'+ZXY] = SBXi[key]['F'+ZXY][:, T1[key, :]]
            del SBXi[key]['F'+ZXY]
    # Output:  inds, indsC, SBXm, SBXi - обрезанные по L
    return SBXm


def save_output(SBXi, SBXm):
    # save output SBXi, SBXm:
    for key in SBXi:
        with h5py.File(f"SBX{key}.h5", 'w') as f:
            group = f.create_group('SBX')
            for sub_key in SBXi[key]:
                group.create_dataset(sub_key, data=SBXi[key][sub_key])
    with h5py.File(f"SBXm.h5", 'w') as f:
        group = f.create_group('SBX')
        for sub_key in SBXm:
            group.create_dataset(sub_key, data=SBXm[sub_key])


def main():
    file_paths = [
        os.sep.join([os.getcwd(), 'SBX_00006.mat']),
        os.sep.join([os.getcwd(), 'SBX_00016.mat']),
    ]
    # file_paths = ['D:/Градиент/RNDnet/Sync/SBX_00006.mat']
    tstrt = 0  # tstrt: начало окна, отсчеты;
    L = 2000  # L: длина окна, отсчеты;
    Lstd = 150  # Lss: длина окна до взрыва для вычисления std, отсчеты;
    dT = [0, 4672]  # dT: сдвиги каждого SBX файла, отсчеты;
    Fpass1 = 20  # Fpass1: начальная частота фильтрации, Гц;
    Fpass2 = 35  # Fpass2: конечная частота фильтрации, Гц;
    mode = 'debugging'  # mode: debugging - режим отладки; unloading - режим выгрузки;
    lev = 0.15  # lev: уровень шума входящего сигнала для отбраковки
    ksen = 91  # nsen: какой канал смотреть, номер (название) канала (датчика, сенсора?)
    extraFUP = True  # применять доп.фильтрацию узкополосных помех checkbox
    df = 5  # ширина медианного фильтра доп.фильтрации узкополосных помех, Гц
    ########################################################################################################################

    SBXi = load_h5_files(file_paths)
    LenOfSignal, fd, acor, b, T1, k, NumCh, nSBX, NamesOfSen, time0 = calc_vars(SBXi, Fpass1, Fpass2, L, tstrt, dT)
    if mode == 'debugging':
        SBX_plot = debugging(SBXi, ksen, extraFUP, LenOfSignal, df, fd, acor, b)
        build_plots(SBXi, SBX_plot, T1, k, file_paths)
    else:
        SBXm = unloading(SBXi, extraFUP, LenOfSignal, df, fd, acor, NumCh, L, nSBX, b, NamesOfSen, Lstd, T1, k, lev, dT, time0)
        # save_output(SBXi, SBXm)


if __name__ == '__main__':
    main()
