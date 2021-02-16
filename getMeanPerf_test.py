# -*- encoding: utf-8 -*-

import os

import numpy as np
import h5py
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication

from filter_f import filter_f
from scipy.signal import medfilt2d
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
from itertools import cycle
from get_set_SBXsen import getSBXsen, setSBXsen

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import rc


def import_h5(file_paths):
    sbx_i = dict()
    for n_sbx, file_path in enumerate(file_paths):
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            # print(keys)
            sbx_f = f.get(keys[0])
            sbx_i[n_sbx] = dict()
            for key in sbx_f:
                sbx_i[n_sbx][key] = np.array(sbx_f[key])
    return sbx_i


class Canvas(FigureCanvas):
    def __init__(self, parent, **kwargs):
        self.__dict__.update(kwargs)
        self.fig, self.axs = plt.subplots(
            nrows=3, ncols=1,  # rows and cols count
            figsize=(1, 1), dpi=180, clear=True,  # dpi - whole mpl graphics scaling
            sharex='all', sharey='all',  # All the plots have same parameters and scale together
            gridspec_kw={
                'hspace': 0, 'wspace': 0,  # Space between plots
                'left': 0.075, 'right': 0.95, 'top': 0.95, 'bottom': 0.05,  # borders
            },
        )
        if self.widget:
            plt.close()
        super(Canvas, self).__init__(self.fig)
        self.setParent(parent)
        for i_xyz, xyz in enumerate(self.titles):
            self.val_count = len(self.funvalues[xyz])
            for i in range(self.val_count):
                self.axs[i_xyz].plot(
                    self.funvalues[xyz][i],
                    label=self.labels[xyz][i],
                    lw=self.lw)
            self.axs[i_xyz].set(
                ylabel=f'{self.titles[i_xyz]}[{self.ylabel}]',
                xlabel=f'[{self.xlabel}]',
                # title=self.titles[i_xyz],
            )
            self.axs[i_xyz].grid()
            # Hide x labels and tick labels for all but bottom plot.
            self.axs[i_xyz].label_outer()
        self.axs[0].legend(loc='upper right', ncol=1)  # self.val_count)
        self.f = self.zoom_factory(self.axs[0], base_scale=1.5)
        # plt.show()

    def show_plots(self):
        plt.show()

    def zoom_factory(self, ax, base_scale=2.):
        """Adds zooming with mouse wheel."""
        def zoom_fun(event):
            # get the current x and y limits
            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location
            if xdata is None or ydata is None:
                return
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1/base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)
            # set new limits
            modifiers = QGuiApplication.keyboardModifiers()
            # If Control Button pressed while scrolling
            if not (modifiers & Qt.ControlModifier):
                ax.set_xlim([xdata - (xdata - cur_xlim[0])*scale_factor,
                             xdata + (cur_xlim[1] - xdata)*scale_factor])
            # If Shift Button pressed while scrolling
            if not (modifiers & Qt.ShiftModifier):
                ax.set_ylim([ydata - (ydata - cur_ylim[0])*scale_factor,
                             ydata + (cur_ylim[1] - ydata)*scale_factor])
            ax.get_figure().canvas.draw_idle()  # force re-draw
        fig = ax.get_figure()  # get the figure of interest
        # attach the call back
        fig.canvas.mpl_connect('scroll_event', zoom_fun)
        #return the function
        return zoom_fun


class SyncMaker(object):
    def __init__(self, **kwargs):
        self.SBXi = {}
        self.file_paths = []
        self.Fpass1 = None
        self.Fpass2 = None
        self.L = None
        self.Lstd = None
        self.mode = None
        self.extraFUP = None
        self.ksen = None
        self.df = None
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

    def make(self, widget=None):
        if widget is not None:
            self.main.clear_layout(widget.layout())
        if self.mode == 'debugging':
            self.get_plots()
            self.build_plots(widget)
        else:
            SBXm = self.unload()
            # save_output(SBXi, SBXm)

    def get_plots(self):
        self.SBX_plot = dict()
        for key in self.SBXi:
            self.SBX_plot[key] = dict()
            for XYZ in ['Z', 'X', 'Y']:
                signal = getSBXsen(self.SBXi[key], 'S'+XYZ, self.ksen[key])
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

    def build_plots(self, widget=None):
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

        # setup font size for chart
        font = {'family': 'Courier New', 'weight': 'bold', 'size': 4.5}
        rc('font', **font)
        labels = {}
        funvalues = {}
        titles = ['FZ', 'FX', 'FY']
        for title in titles:
            labels[title] = []
            funvalues[title] = []
            for key in self.SBXi:
                labels[title].append(os.path.basename(self.file_paths[key]))
                funvalues[title].append(self.SBX_plot[key][title][self.T1[key, :]] * self.k[key])
        kwargs = {
            'xlabel': 'Time',
            'ylabel': 'Amplitude',
            'labels': labels,
            'funvalues': funvalues,

            'lw': 0.8,
            'titles': titles,
            'widget': widget,
        }
        chart = Canvas(parent=widget, **kwargs)
        chart.setObjectName(f'{title}chart')
        if widget is None:
            chart.show_plots()
        else:
            self.main.ui.__dict__[f'{title}chart'] = chart
            toolbar = NavigationToolbar(self.main.ui.__dict__[f'{title}chart'], widget)
            self.main.ui.__dict__[f'{title}toolbar'] = toolbar
            toolbar.setObjectName(f'{title}toolbar')
            # Turn on pan/zoom mode from start
            toolbar.pan()
            widget.layout().addWidget(toolbar)
            widget.layout().addWidget(chart)

    def unload(self):
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
        'ksen': [91, 91],  # ksen: какой канал смотреть, номер (название) канала (датчика, сенсора?)
        'extraFUP': True,  # применять доп.фильтрацию узкополосных помех checkbox
        'df': 5,  # ширина медианного фильтра доп.фильтрации узкополосных помех, Гц
    }
    ########################################################################################################################
    sync_maker = SyncMaker(**kwargs)
    sync_maker.make()


if __name__ == '__main__':
    main()
