# -*- coding: utf-8 -*-
import h5py
from PyQt5.QtCore import QAbstractListModel, Qt, QSize, QAbstractItemModel
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QMessageBox, QDoubleSpinBox, QLabel

from SyncGraph import Ui_syncGraphMainWindow  # Graphical user interface
from matplotlib import pyplot as plt
import os
import json
import numpy as np

SET_F_NAME = 'settings.json'


def import_h5(file_paths):
    """Import one or plural h5 files
    Args:
        file_paths (list): h5 files locations
    Return:
        sbx_i (dict): data of files
    """
    sbx_i = dict()
    for n_sbx, file_path in enumerate(file_paths):
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            sbx_f = f.get(keys[0])
            sbx_i[n_sbx] = dict()
            for key in sbx_f:
                sbx_i[n_sbx][key] = np.array(sbx_f[key])
    return sbx_i


class InFilesList(QAbstractListModel):
    """Inner files list synchronization model"""

    def __init__(self, *args, paths=None, **kwargs):
        super(InFilesList, self).__init__(*args, **kwargs)
        self.paths = paths or []

    def data(self, index, role):
        status, text = self.paths[index.row()]
        if role == Qt.DisplayRole:
            # Return the name only
            return os.path.basename(text)
        if role == Qt.ToolTipRole:
            # Return full path
            return text
        if role == Qt.CheckStateRole:
            return status

    def rowCount(self, index):
        return len(self.paths)


class SensorsList(QAbstractListModel):
    """List of sensors synchronization model"""

    def __init__(self, *args, ids=None, **kwargs):
        super(SensorsList, self).__init__(*args, **kwargs)
        self.ids = ids or []

    def data(self, index, role):
        text = self.ids[index.row()]  # [0]
        if role == Qt.DisplayRole:
            # Return the name only
            return text

    def rowCount(self, index):
        return len(self.ids)


class CalculationsRunner():
    """Value synchronizes with visual component"""
    def __init__(self, *args, main, **kwargs):
        self.main = main
        main.ui.tstrtEdit.textChanged.connect(lambda: self.autochange('tstrt', main.ui.tstrtEdit))
        main.ui.LEdit.textChanged.connect(lambda: self.autochange('L', main.ui.LEdit))
        main.ui.LstdEdit.textChanged.connect(lambda: self.autochange('Lstd', main.ui.LstdEdit))
        main.ui.Fpass1Edit.textChanged.connect(lambda: self.autochange('Fpass1', main.ui.Fpass1Edit))
        main.ui.Fpass2Edit.textChanged.connect(lambda: self.autochange('Fpass2', main.ui.Fpass2Edit))
        main.ui.dfEdit.textChanged.connect(lambda: self.autochange('df', main.ui.dfEdit))
        main.ui.levEdit.textChanged.connect(lambda: self.autochange('lev', main.ui.levEdit))
        main.ui.tstrtEdit.textChanged.connect(lambda: self.autochange('tstrt', main.ui.tstrtEdit))
        # self.tstrt_autochange()
        # self.L_autochange()
        # self.Lstd_autochange()
        # self.Fpass1_autochange()
        # self.Fpass2_autochange()
        # self.df_autochange()
        # self.lev_autochange()
        if main.ui.unloadingRadioButton.isChecked():  # 0 - debugging, 1 - unloading
            self.mode = 'unloading'
        elif main.ui.debuggingRadioButton.isChecked():
            self.mode = 'debugging'
        self.extraFUP = main.ui.extraFUPCheckBox.isChecked()
        self.nsen = 0
        main.ui.sensorsListView.clicked.connect(self.choose_nsen)
        # self.dT = or []
        print(self.nsen)
        for child in self.main.ui.shiftsScrollArea.widget().children():
            if isinstance(child, QDoubleSpinBox):
                child.textChanged.connect(lambda: self.dT_autochange(number=child.value()))

    def choose_nsen(self):
        try:
            selected_index = self.main.ui.sensorsListView.selectedIndexes()[0].row()
        except Exception:
            print('Could not get sensor number')
            return
        self.nsen = self.main.ksen[f'{selected_index:05.0f}']
        print(self.nsen)

    def autochange(self, variable_name, field):
        self.__dict__[variable_name] = field.value()

class MainWindow(QMainWindow):
    """The main window of the application."""

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.state_ui = dict()
        # Load user interface generated in QtDesigner
        self.ui = Ui_syncGraphMainWindow()
        self.ui.setupUi(self)
        # Create in_files model and connect it to listview
        self.in_files = InFilesList()
        self.sbx_files = None  # Will load h5 files in this variable
        self.sensors = None  # Will load sensors names in this variable
        self.ui.inFilesListView.setModel(self.in_files)
        # Buttons and other objects methods
        self.ui.loadButton.clicked.connect(self.load_h5_files)
        self.ui.addInFileButton.clicked.connect(self.add_in_files)
        self.ui.removeInFileButton.clicked.connect(self.delete_in_files)
        self.ui.checkFilesButton.clicked.connect(self.change_in_files_status)
        # Import Window state from settings file
        self.import_state()
        self.calculations = None

    def populate_sensors(self):
        if len(self.sbx_files) > 0:
            self.ksen = dict()
            for file_id, sbx_file in enumerate(self.sbx_files.items()):
                for i, x in enumerate(sbx_file[1]['field'][0]):
                    s_name = f'{x:05.0f}'
                    if s_name not in self.ksen:
                        self.ksen[s_name] = [None for x in range(file_id)]
                    self.ksen[f'{s_name}'].append(i)
                for i in self.ksen.values():
                    if len(i) < file_id + 1:
                        i.append(None)

            self.sensors = SensorsList(ids=sorted(self.ksen))
            self.ui.sensorsListView.setModel(self.sensors)

    def load_h5_files(self):
        self.in_files.checked_list = [x[1] for x in self.in_files.paths if x[0]]
        try:
            self.sbx_files = import_h5(self.in_files.checked_list)
        except Exception:
            print(f'Exception:{Exception}\n'
                  "Couldn't read a file.")
            return
        layout = self.ui.shiftsScrollArea.widget().layout()
        self.clear_layout(layout)
        self.populate_shifts(layout)
        self.populate_sensors()
        self.calculations = CalculationsRunner(main=self)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def populate_shifts(self, layout):
        for i_sbx, sbx_file in enumerate(self.sbx_files):
            label = QLabel(layout.parent())
            label.setObjectName(f"ShiftLabel_{i_sbx}")
            label.setText(f"File_{i_sbx}: {os.path.basename(self.in_files.checked_list[i_sbx])}")
            edit_field = QDoubleSpinBox(layout.parent())
            edit_field.setObjectName(f"ShiftEdit_{i_sbx}")
            layout.addWidget(label, i_sbx, 0, 1, 1)
            layout.addWidget(edit_field, i_sbx, 1, 1, 1)

    def change_in_files_status(self):
        indexes = self.ui.inFilesListView.selectedIndexes()
        if indexes:
            value = False
            for index in indexes:
                if not self.in_files.paths[index.row()][0]:
                    value = True
                    break
            for index in indexes:
                self.in_files.paths[index.row()][0] = value
            self.in_files.layoutChanged.emit()

    def add_in_files(self):
        filenames, filetype = QFileDialog.getOpenFileNames(
            self, 'Open inner data file', os.getcwd(), 'HDF5 data files (*.mat *.h5)')
        for filename in filenames:
            # Don't add empty strings and existing files.
            if filename and (filename not in [x[1] for x in self.in_files.paths]):
                # Access the list via the model.
                self.in_files.paths.append([False, filename])
        # Trigger refresh.
        self.in_files.layoutChanged.emit()

    def delete_in_files(self):
        if QMessageBox.question(
                self, 'Question', 'Are you sure you want to delete items from the list?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) == QMessageBox.No:
            return
        indexes = self.ui.inFilesListView.selectedIndexes()
        if indexes:
            # Indexes is a list of a single item in single-select mode.
            for index in reversed(indexes):
                # Remove the item
                del self.in_files.paths[index.row()]
            # Refresh after all
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
