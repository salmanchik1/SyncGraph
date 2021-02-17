# -*- coding: utf-8 -*-
import time
from functools import partial

import h5py
from PyQt5.QtCore import QAbstractListModel, Qt, QSize, QAbstractItemModel
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QMessageBox, QDoubleSpinBox, QLabel
from getMeanPerf_test import SyncMaker, import_h5
from SyncGraph import Ui_syncGraphMainWindow  # Graphical user interface
from matplotlib import pyplot as plt
import os
import json
import numpy as np

SET_F_NAME = 'settings.json'


class InFilesModel(QAbstractListModel):
    """Inner files list synchronization model"""

    def __init__(self, *args, paths=None, **kwargs):
        super(InFilesModel, self).__init__(*args, **kwargs)
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


class SensorsModel(QAbstractListModel):
    """List of sensors synchronization model"""

    def __init__(self, *args, ids=None, **kwargs):
        super(SensorsModel, self).__init__(*args, **kwargs)
        self.ids = ids or []

    def data(self, index, role):
        text = self.ids[index.row()]  # [0]
        if role == Qt.DisplayRole:
            # Return the name only
            return text

    def rowCount(self, index):
        return len(self.ids)


class SyncMakerGraph(SyncMaker):
    """Value synchronizes with visual component"""

    def __init__(self,  main_window=None):
        self.selected_s_name = None
        if main_window is None:
            print('Main caller class not founded.')
            return
        else:
            self.main = main_window
        var_names = ['tstrt', 'L', 'Lstd', 'Fpass1', 'Fpass2', 'df', 'lev']
        self.sync_vars(var_names)
        self.sync_dTs()
        self.main.ui.sensorsListView.clicked.connect(self.on_select_sensor)
        self.extraFUP = self.main.ui.extraFUPCheckBox.isChecked()
        self.main.ui.extraFUPCheckBox.clicked.connect(
            partial(self.on_click_checkbox, variable_name='extraFUP'))
        self.main.ui.debuggingRadioButton.clicked.connect(self.on_click_mode)  # 0 - debugging, 1 - unloading
        self.main.ui.unloadingRadioButton.clicked.connect(self.on_click_mode)  # 0 - debugging, 1 - unloading
        self.on_click_mode()
        self.on_select_sensor()
        self.SBXi = {i: x for i, x in enumerate(self.main.checked_files.values())}
        # self.ksen = [0 for i in file_paths]
        kwargs = {
            'file_paths': [*self.main.checked_files.keys()],  # a list of directories
            'SBXi': self.SBXi,  # SBXi: файлы
            'tstrt': self.tstrt,  # tstrt: начало окна, отсчеты;
            'L': self.L,  # L: длина окна, отсчеты;
            'Lstd': self.Lstd,  # Lss: длина окна до взрыва для вычисления std, отсчеты;
            'dT': self.dT,  # dT: сдвиги каждого SBX файла, отсчеты;
            'Fpass1': self.Fpass1,  # Fpass1: начальная частота фильтрации, Гц;
            'Fpass2': self.Fpass2,  # Fpass2: конечная частота фильтрации, Гц;
            'mode': self.mode,  # mode: debugging - режим отладки; unloading - режим выгрузки;
            'lev': self.lev,  # lev: уровень шума входящего сигнала для отбраковки
            'ksen': self.ksen,  # ksen: какой канал смотреть, номер (название) канала (датчика, сенсора?)
            'extraFUP': self.extraFUP,  # применять доп.фильтрацию узкополосных помех checkbox
            'df': self.df,  # ширина медианного фильтра доп.фильтрации узкополосных помех, Гц
        }
        super().__init__(**kwargs)
        # self.make()

    def on_click_build(self):
        self.make_build(widget=self.main.ui.graphsContainer)

    def sync_vars(self, vars):
        """Synchronizes variables"""
        # All the fields values being saved to variables by autochange function
        for var in vars:
            # self.__dict__[var] = self.main.ui.__dict__[f'{var}Edit'].value()
            self.main.ui.__dict__[f'{var}Edit'].textChanged.connect(
                partial(self.on_change_edit, variable_name=var))
            self.on_change_edit(variable_name=var)

    def sync_dTs(self):
        self.dT = []
        i_variable = 0
        for child in self.main.ui.shiftsScrollArea.widget().children():
            if isinstance(child, QDoubleSpinBox):
                self.dT.append(f'dT{i_variable}')
                self.main.ui.__dict__[f'dT{i_variable}Edit'].textChanged.connect(
                    partial(self.on_click_dT, index=i_variable))
                self.on_click_dT(index=i_variable)
                i_variable += 1

    def on_click_dT(self, index):
        self.dT[index] = self.main.ui.__dict__[f'dT{index}Edit'].value()
        print(f'dT{index} = {self.__dict__["dT"][index]}')

    def on_change_edit(self, variable_name):
        # I take the value from the gui widget and put it in the variable named the same
        self.__dict__[variable_name] = self.main.ui.__dict__[f'{variable_name}Edit'].value()
        print(f'{variable_name} = {self.__dict__[f"{variable_name}"]}')

    def on_select_sensor(self):
        try:
            selected_index = self.main.ui.sensorsListView.selectedIndexes()[0].row() + 1
        except Exception:
            selected_index = 1
        self.selected_s_name = f'{selected_index:05.0f}'
        self.ksen = self.main.s_names[self.selected_s_name]
        # self.make(widget=self.main.ui.graphsContainer)
        print(self.ksen)

    def on_click_checkbox(self, variable_name):
        self.__dict__[variable_name] = self.main.ui.__dict__[variable_name + 'CheckBox'].isChecked()
        print(f'{variable_name} = {self.__dict__[variable_name]}')

    def on_click_mode(self):
        if self.main.ui.debuggingRadioButton.isChecked():
            self.mode = 'debugging'
        elif self.main.ui.unloadingRadioButton.isChecked():
            self.mode = 'unloading'
        print(f'mode = {self.mode}')

    def reload(self, **kwargs):
        self.__dict__.update(kwargs)
        self.sync_dTs()
        var_names = ['tstrt', 'L', 'Lstd', 'Fpass1', 'Fpass2', 'df', 'lev']
        self.sync_vars(var_names)
        self.on_select_sensor()


class MainWindow(QMainWindow):
    """The main window of the application."""

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.state_ui = dict()
        # Load user interface generated in QtDesigner
        self.ui = Ui_syncGraphMainWindow()
        self.ui.setupUi(self)
        # Create in_files model and connect it to listview
        self.in_files = InFilesModel()
        self.sbx_files = None  # Will load h5 files in this variable
        self.checked_files = None
        self.sensors = None  # Will load sensors names in this variable
        self.ui.inFilesListView.setModel(self.in_files)
        # Buttons and other objects methods
        self.ui.addInFileButton.clicked.connect(self.add_in_files)
        self.ui.loadButton.clicked.connect(self.on_click_load)
        self.ui.removeInFileButton.clicked.connect(self.delete_in_files)
        self.ui.inFilesListView.mouseReleaseEvent = lambda x: self.on_select_in_files(x)
        # Import Window state from settings file
        self.import_state()
        self.sync_maker = None
        self.refresh_in_files()

    def refresh_in_files(self):
        if self.sbx_files is None:
            self.sbx_files = dict()
        for path in self.in_files.paths:
            if path[1] in self.sbx_files:
                continue
            try:
                self.sbx_files[path[1]] = import_h5(path[1])
            except Exception as e:
                print(f'Exception:{e}\n'
                      "File reading error or it doesn't exist.")
        # Clear out redundant items from sbx_files (free up the memory)
        delete = [key for key in self.sbx_files.keys() if key not in
                  [x[1] for x in self.in_files.paths]]
        for key in delete:
            del self.sbx_files[key]

    def on_click_load(self):
        self.checked_files = dict()
        for item in self.sbx_files.items():
            if [True, item[0]] in self.in_files.paths:
                self.checked_files[item[0]] = item[1]
        layout = self.ui.shiftsScrollArea.widget().layout()
        self.clear_layout(layout)
        self.fill_shifts(layout)
        self.fill_sensors()
        if self.sync_maker is None:
            self.sync_maker = SyncMakerGraph(main_window=self)
        else:
            kwargs = {
                'SBXi': {i: x for i, x in enumerate(self.checked_files.values())},
                'file_paths': [*self.checked_files.keys()]
            }
            self.sync_maker.reload(**kwargs)
        self.ui.buildButton.clicked.connect(self.sync_maker.on_click_build)

    def fill_sensors(self):
        if len(self.checked_files) > 0:
            self.s_names = dict()
            for file_id, _file in enumerate(self.checked_files.items()):
                for i, x in enumerate(_file[1]['field'][0]):
                    s_name = f'{x:05.0f}'
                    if s_name not in self.s_names:
                        self.s_names[s_name] = [None for _ in range(file_id-1)]
                    self.s_names[s_name].append(i + 1)
            for i in self.s_names.values():
                if len(i) < len(self.checked_files):
                    i.append(None)

            self.sensors = SensorsModel(ids=sorted(self.s_names))
            self.ui.sensorsListView.setModel(self.sensors)
        else:
            if self.sensors is not None:
                # clear out all the sensors data
                self.sensors = SensorsModel(ids=[])
                self.ui.sensorsListView.setModel(self.sensors)

    def clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def fill_shifts(self, layout):
        for i_sbx, filename in enumerate(self.checked_files):
            label = QLabel(layout.parent())
            label.setObjectName(f"dT{i_sbx}Label")
            label.setText(f"File_{i_sbx}: {os.path.basename(filename)}")
            edit_field = QDoubleSpinBox(layout.parent())
            edit_field.setObjectName(f"dT{i_sbx}Edit")
            edit_field.setMinimum(-65000)
            edit_field.setMaximum(65000)
            edit_field.setValue(i_sbx*5)
            self.ui.__dict__[f'dT{i_sbx}Label'] = label
            self.ui.__dict__[f'dT{i_sbx}Edit'] = edit_field
            layout.addWidget(label, i_sbx, 0, 1, 1)
            layout.addWidget(edit_field, i_sbx, 1, 1, 1)

    def on_select_in_files(self, e=None):
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
            self.ui.inFilesListView.reset()

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
        self.refresh_in_files()

    def delete_in_files(self):
        if QMessageBox.question(
                self, 'Question', 'Are you sure want to delete checked items from the list?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) == QMessageBox.No:
            return
        indexes = [i for i, x in enumerate(self.in_files.paths) if x[0]]
        if indexes:
            # Indexes is a list of a single item in single-select mode.
            for index in reversed(indexes):
                # Remove the item
                del self.in_files.paths[index]
            # Refresh after all
            self.in_files.layoutChanged.emit()
            # Clear the selection (as it is no longer valid).
            self.ui.inFilesListView.clearSelection()
        self.refresh_in_files()

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

    def closeEvent(self, event):
        """Raises on main window closing"""
        self.export_state()
        event.accept()


def main():
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
