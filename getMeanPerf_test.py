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


def import_h5(file_paths):
    sbx_i = dict()
    for n_sbx, file_path in enumerate(file_paths):
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            print(keys)
            sbx_f = f.get(keys[0])
            sbx_i[n_sbx] = dict()
            for key in sbx_f:
                sbx_i[n_sbx][key] = np.array(sbx_f[key])
    return sbx_i


class SyncMaker(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.nSBX = len(self.SBXi)
        key = list(self.SBXi.keys())[0]
        self.LenOfSignal = self.SBXi[key]['SZ'].shape[1]
        # создаем множество set() имен сенсоров NamesOfSen из всех входных данных
        self.NamesOfSen = set()
        for key in self.SBXi:
            self.NamesOfSen.update(self.SBXi[key]['field'][0, :])

        self.fd = self.SBXi[key]['fd'][0][0]  # частота дискретизации сигнала
        self.b = filter_f(self.fd, self.Fpass1, self.Fpass2)
        self.tstrt = self.tstrt / self.fd
        self.time0 = np.arange(self.L, dtype=int) + self.tstrt * self.fd
        self.T1 = np.zeros((self.nSBX, self.L), dtype=int)
        for i in range(self.nSBX):
            self.T1[i, :] = self.dT[i] + self.time0
        self.T1[self.T1 >= self.LenOfSignal] -= self.LenOfSignal
        self.k = np.ones(self.nSBX) * 1000000 * 1.91037945231066 * (10 ** -8) / self.fd
        self.acor = 1  # неизвестная константа?
        self.NumCh = len(self.NamesOfSen)

    def make(self):
        if self.mode == 'debugging':
            self.debugging()
            self.build_plots()
        else:
            SBXm = self.unloading()
            # save_output(SBXi, SBXm)

    def debugging(self):
        self.SBX_plot = dict()
        for key in self.SBXi:
            self.SBX_plot[key] = dict()
            for XYZ in ['Z', 'X', 'Y']:
                signal = getSBXsen(self.SBXi[key], 'S'+XYZ, self.ksen)
                if self.extraFUP:
                    ldf = int(self.LenOfSignal * self.df / self.fd)
                    n_fup = ldf if ldf else ldf - 1
                    sp = np.fft.fft(signal)
                    norsp = abs(sp)
                    # medsp = medfilt(norsp, n_fup)
                    medsp = medfilt2d(norsp.reshape(1, -1), (1, n_fup))[0]  # на порядок быстрее medfilt
                    signal = self.acor * (np.fft.ifft((sp / norsp) * medsp)).real
                # SBX_plot[key]['F' + XYZ] = lfilter(b, 1, signal)[:, int((len(b) - 1)/2):]
                self.SBX_plot[key]['F' + XYZ] = filtfilt(self.b, 1, signal)

    def build_plots(self):
        # for key in self.SBXi:
        #     for FZXY in ['FZ', 'FX', 'FY']:
        #         fig, ax = plt.subplots()
        #         ax.plot(SBX_plot[key][FZXY][self.T1[key, :]] * \
        #             self.k[key], label=file_paths[key])
        #         # maxindex = np.argmax(SBX_plot[key][FZXY][T1[key, :]])
        #         # plt.axvline(
        #         #   x=maxindex, label='MAX line at {}'.format(maxindex), c='r')
        #         plt.title(FZXY)
        #         ax.grid()
        #         ax.legend(loc='upper right', ncol=2)
        #         plt.show()
    
        for FZXY in ['FZ', 'FX', 'FY']:
            fig, ax = plt.subplots()
            for key in self.SBXi:
                ax.plot(
                    self.SBX_plot[key][FZXY][self.T1[key, :]] * self.k[key], 
                    label=self.file_paths[key], lw=0.8)
            plt.title(FZXY)
            ax.grid()
            ax.legend(loc='upper right', ncol=2)
            plt.show()
    
    def unloading(self):
        for key in self.SBXi:
            for XYZ in ['Z', 'X', 'Y']:
                if self.extraFUP:
                    ldf = int(self.LenOfSignal * self.df / self.fd)
                    n_fup = ldf if ldf else ldf - 1
                    for ksen in self.SBXi[key]['field'][0, :]:
                        print(ksen)
                        sp = np.fft.fft(getSBXsen(self.SBXi[key], 'S' + XYZ, ksen))
                        norsp = abs(sp)
                        # medsp = medfilt(norsp, n_fup)
                        medsp = medfilt2d(norsp.reshape(1, -1), (1, n_fup))[0]  # на порядок быстрее medfilt
                        setSBXsen(self.SBXi[key], 'S' + XYZ, ksen, self.acor * (np.fft.ifft((sp / norsp) * medsp)).real)
                # self.SBXi[key]['F'+XYZ] = lfilter(self.b, 1, self.SBXi[key]['S'+XYZ])[:, int((len(self.b) - 1)/2):]
                self.SBXi[key]['F' + XYZ] = filtfilt(self.b, 1, self.SBXi[key]['S' + XYZ])
        # создаем усредненный SBX
        field = np.zeros((4, len(self.NamesOfSen)))
        field[0, :] = np.array(list(self.NamesOfSen))
        self.SBXm = {
            'SZ': np.full((self.NumCh, self.L), np.nan),
            'SX': np.full((self.NumCh, self.L), np.nan),
            'SY': np.full((self.NumCh, self.L), np.nan),
            'fd': self.fd,
            # заполняем field для него из всех входных SBX( field может отличаться, координаты для одинаковых сенсоров тоже)
            'field': field,}
        keys = cycle(list(self.SBXi.keys()))
        key = next(keys)
        for i in range(len(self.NamesOfSen)):
            while self.SBXm['field'][0, i] not in self.SBXi[key]['field'][0, :]:
                key = next(keys)
            index = np.where(self.SBXi[key]['field'][0, :] == self.SBXm['field'][0, i])[0][0]
            self.SBXm['field'][:, i] = self.SBXi[key]['field'][:, index]
    
        inds = np.empty((self.NumCh * 3, self.nSBX), dtype=int)
        indsC = np.empty(self.NumCh * 3, dtype=int)
    
        for ch in self.NamesOfSen:
            jj = 0
            # fig, axs = plt.subplots(3, 1, constrained_layout=True)
            # fig.suptitle('Channel number: ' + str(int(SBXi[0]['field'][0, ch])), fontsize=16)
    
            for ZXY in ['Z', 'X', 'Y']:
                Ss = np.full((self.nSBX, self.L), np.nan)    # np.full((2, 3), np.nan)
                Sstd = np.full((self.nSBX, self.Lstd), np.nan)
                for i in range(self.nSBX):
                    Ss[i, :] = getSBXsen(self.SBXi[i], 'F'+ZXY, ch)[self.T1[i, :]]*self.k[i]
                    Sstd[i, :] = getSBXsen(self.SBXi[i], 'F' + ZXY, ch)[self.T1[i, 0:self.Lstd]] * self.k[i]
                std = np.std(Sstd, axis=1)
                I = std.argsort(axis=0)
                std.sort(axis=0)
                I = np.concatenate((I[std != 0], I[std == 0]))
                inds[int((ch-1)*3+jj), :] = I
                I = I[std < self.lev]
                indsC[int((ch-1)*3+jj)] = len(I)
    
                # plot ?
                # axs[jj].plot(np.mean(Ss[I, :], axis=0)*len(I))
                # axs[jj].set_title(ZXY)
                # axs[jj].axis([0, L, -1, 1])
    
                setSBXsen(self.SBXm, 'S'+ZXY, ch, np.mean(Ss[I, :], axis=0))
                jj += 1
    
            # plt.show()
    
        self.dT = self.dT + self.time0[0]
        self.SBXm['SZ'][np.isnan(self.SBXm['SZ'])] = 0
        self.SBXm['SX'][np.isnan(self.SBXm['SX'])] = 0
        self.SBXm['SY'][np.isnan(self.SBXm['SY'])] = 0
    
        for key in self.SBXi:
            for ZXY in ['Z', 'X', 'Y']:
                self.SBXi[key]['S'+ZXY] = self.SBXi[key]['F'+ZXY][:, self.T1[key, :]]
                del self.SBXi[key]['F'+ZXY]
        # Output:  inds, indsC, SBXm, SBXi - обрезанные по L

    def save_output(self, SBXi, SBXm):
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
    kwargs = {
        'file_paths': file_paths,  # a list of directories
        'SBXi': import_h5(file_paths),  # SBXi: файлы
        'tstrt': 0,  # tstrt: начало окна, отсчеты;
        'L': 2000,  # L: длина окна, отсчеты;
        'Lstd': 150,  # Lss: длина окна до взрыва для вычисления std, отсчеты;
        'dT': [0, 4672],  # dT: сдвиги каждого SBX файла, отсчеты;
        'Fpass1': 20,  # Fpass1: начальная частота фильтрации, Гц;
        'Fpass2': 35,  # Fpass2: конечная частота фильтрации, Гц;
        'mode': 'debugging',  # mode: debugging - режим отладки; unloading - режим выгрузки;
        'lev': 0.15,  # lev: уровень шума входящего сигнала для отбраковки
        'ksen': 91, # nsen: какой канал смотреть, номер (название) канала (датчика, сенсора?)
        'extraFUP': True,  # применять доп.фильтрацию узкополосных помех checkbox
        'df': 5,  # ширина медианного фильтра доп.фильтрации узкополосных помех, Гц
    }
    ########################################################################################################################
    sync_maker = SyncMaker(**kwargs)
    sync_maker.make()


if __name__ == '__main__':
    main()
