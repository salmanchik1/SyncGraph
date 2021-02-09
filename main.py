# -*- coding: utf-8 -*-



from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication

from SyncGraph import Ui_syncGraphMainWindow
from matplotlib import pyplot as plt
import os


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load user interface generated in QtDesigner
        self.ui = Ui_syncGraphMainWindow()
        self.ui.setupUi(self)

        self.ui.inFileChooseButton.clicked.connect(self._choose_in_path)
        self._refresh_in_files()

    def _refresh_in_files(self):
        self.ui.inFilePathEdit.setText(in_files_path)

    def _choose_in_path(self):
        global in_files_path
        in_files_path = QFileDialog.getExistingDirectory(self, "Select inner files location", QDir.currentPath())
        self._refresh_in_files()


if __name__ == '__main__':
    import sys
    # define global variables
    in_files_path = os.getcwd()

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

