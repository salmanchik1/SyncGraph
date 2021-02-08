# -*- coding: utf-8 -*-


from PyQt5 import QtCore, QtGui, QtWidgets
from SyncGraph import Ui_syncGraphMainWindow
from matplotlib import pyplot as plt


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load user interface generated in QtDesigner
        self.ui = Ui_syncGraphMainWindow()
        self.ui.setupUi(self)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

