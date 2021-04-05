# -*- coding: utf-8 -*-
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Canvas(FigureCanvas):
    def __init__(self, parent, **kwargs):
        self.__dict__.update(kwargs)
        # plt.ion()
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
        self.ticks_changed = True
        self.setParent(parent)
        self.val_count = len(self.funvalues[self.titles[0]])
        for i_xyz, xyz in enumerate(self.titles):
            for i in range(self.val_count):
                if self.ticks_mode == 'seconds':
                    xvalues = [i / self.fd for i in range(len(self.funvalues[xyz][i]))]
                else:
                    xvalues = range(len(self.funvalues[xyz][i]))
                self.axs[i_xyz].plot(
                    xvalues,
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
            # self.axs[i_xyz].get_xaxis().get_major_formatter().set_useOffset(False)
            self.axs[i_xyz].fmt_xdata = lambda x: f'{x:.0f}'
            self.axs[i_xyz].fmt_ydata = lambda x: f'{x:.3f}'
        self.fig.suptitle(self.suptitle)
        self.axs[0].legend(loc='upper right', ncol=1)  # self.val_count)
        self.zoom_factory(self.axs[0], base_scale=1.1)
        self.window_factory(self.axs[0])
        # plt.show()

    def reload(self, **kwargs):
        self.__dict__.update(kwargs)
        self.xlim = self.axs[0].get_xlim()
        self.ylim = self.axs[0].get_ylim()
        self.val_count = len(self.funvalues[self.titles[0]])
        for i_xyz, xyz in enumerate(self.titles):
            self.axs[i_xyz].clear()
            for i in range(self.val_count):
                if self.ticks_mode == 'seconds':
                    xvalues = [x / self.fd for x in range(len(self.funvalues[xyz][i]))]
                else:
                    xvalues = range(len(self.funvalues[xyz][i]))
                self.axs[i_xyz].plot(
                    xvalues,
                    self.funvalues[xyz][i],
                    label=self.labels[xyz][i],
                    lw=self.lw)
            prop = {
                'ylabel': f'{self.titles[i_xyz]}[{self.ylabel}]',
                'xlabel': f'[{self.xlabel}]',
                # title=self.titles[i_xyz],
            }
            self.axs[i_xyz].grid()
            # Hide x labels and tick labels for all but bottom plot.
            self.axs[i_xyz].label_outer()
            self.axs[i_xyz].update(prop)
        self.fig.suptitle(self.suptitle)
        self.axs[0].legend(loc='upper right', ncol=1)  # self.val_count)
        if not self.ticks_changed:
            self.ticks_changed = True
            if self.ticks_mode == 'seconds':
                self.xlim /= self.fd
            else:
                self.xlim = (x * self.fd for x in self.xlim)
        self.axs[0].set_xlim(self.xlim)
        self.axs[0].set_ylim(self.ylim)
        self.fig.canvas.draw()

    @staticmethod
    def show_plots():
        plt.show()

    @staticmethod
    def zoom_factory(ax, base_scale=2.):
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

    @staticmethod
    def window_factory(ax):
        """Gets window by mouse clicks"""
        def get_window(event):
            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location
            pass
        fig = ax.get_figure()  # get the figure of interest
        # attach the call back
        fig.canvas.mpl_connect('scroll_event', get_window)
        #return the function
        return get_window


