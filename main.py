# -*- coding: utf-8 -*-
from PyQt5.QtCore import QDir, QAbstractListModel, Qt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication

from SyncGraph import Ui_syncGraphMainWindow  # Graphical user interface
from matplotlib import pyplot as plt
import os
import json


SET_F_NAME = 'settings.json'


class InFilesList(QAbstractListModel):
    """Inner files list synchronization model"""
    def __init__(self, *args, paths=None, **kwargs):
        super(InFilesList, self).__init__(*args, **kwargs)
        self.paths = paths or []

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the data structure.
            status, text = self.paths[index.row()]
            # Return the paths text only
            return text

    def rowCount(self, index):
        return len(self.paths)


class MainWindow(QMainWindow):
    """The main window of the application."""
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.state_ui = dict()
        # Load user interface generated in QtDesigner
        self.ui = Ui_syncGraphMainWindow()
        self.ui.setupUi(self)
        # Import Window state from settings file
        self.in_files = InFilesList()
        self.ui.inFilesListView.setModel(self.in_files)
        self.ui.addInFileButton.clicked.connect(self.add_in_files)
        self.ui.removeInFileButton.clicked.connect(self.delete_in_files)
        self.import_state()

    def add_in_files(self):
        filenames, filetype = QFileDialog.getOpenFileNames(
            self, 'Open inner data file', os.getcwd(), 'HDF5 data files (*.mat *.h5)')
        for filename in filenames:
            # Don't add empty strings and existing files.
            if filename and (filename not in [x[1] for x in self.in_files.paths]):
                # Access the list via the model.
                self.in_files.paths.append((False, filename))
                # Trigger refresh.
                self.in_files.layoutChanged.emit()

    def delete_in_files(self):
        indexes = self.ui.inFilesListView.selectedIndexes()
        if indexes:
            # Indexes is a list of a single item in single-select mode.
            for index in reversed(indexes):
                # Remove the item and refresh.
                del self.in_files.paths[index.row()]
            self.in_files.layoutChanged.emit()
            # Clear the selection (as it is no longer valid).
            self.ui.inFilesListView.clearSelection()

    def closeEvent(self, event):
        """Raises on main window closing"""
        self.export_state()
        event.accept()

    def refresh_in_files_ui(self):
        for i in self.state_ui['in_files']:
            self.ui.inFilesListView.append()
        self.ui.inFilePathEdit.setText(self.state_ui['in_file_path'])

    def import_state(self):
        if not os.path.exists(SET_F_NAME):
            return False
        with open(SET_F_NAME, mode='r', encoding='utf-8') as f:
            self.state_ui = json.load(f)
        if 'paths' in self.state_ui.keys():
            self.in_files.paths = self.state_ui['paths']

    def export_state(self):
        self.state_ui['paths'] = self.in_files.paths
        with open(SET_F_NAME, mode='w', encoding='utf-8') as f:
            json.dump(self.state_ui, f, indent=2)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
