# -*- coding: utf-8 -*-


from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication

from SyncGraph import Ui_syncGraphMainWindow
from matplotlib import pyplot as plt
import os
import json


SET_F_NAME = 'settings.json'


class MainWindow(QMainWindow):
    """The main window of the application."""
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.state_ui = dict()
        # Load user interface generated in QtDesigner
        self.ui = Ui_syncGraphMainWindow()
        self.ui.setupUi(self)
        # Import Window state from settings file
        self.import_state()

        self.ui.inFileChooseButton.clicked.connect(self.choose_in_path)
        self.refresh_in_files_ui()

    def closeEvent(self, event):
        self.export_state()
        event.accept()

    def refresh_in_files_ui(self):
        self.ui.inFilePathEdit.setText(in_file_path)

    def choose_in_path(self):
        global in_file_path
        in_file_path = QFileDialog.getExistingDirectory(self, "Select inner files location", QDir.currentPath())
        self.refresh_in_files_ui()

    def import_state(self):
        if not os.path.exists(SET_F_NAME):
            return False
        with open(SET_F_NAME, mode='r', encoding='utf-8') as f:
            self.state_ui = json.load(f)

    def export_state(self):
        self.state_ui['a'] = 'q3e'
        with open(SET_F_NAME, mode='w', encoding='utf-8') as f:
            json.dump(self.state_ui, f, indent=2)




if __name__ == '__main__':
    import sys

    # define global variables
    in_file_path = os.getcwd()

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
