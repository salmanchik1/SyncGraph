# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SyncGraph.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_syncGraphMainWindow(object):
    def setupUi(self, syncGraphMainWindow):
        syncGraphMainWindow.setObjectName("syncGraphMainWindow")
        syncGraphMainWindow.resize(1024, 768)
        syncGraphMainWindow.setMinimumSize(QtCore.QSize(1024, 768))
        self.centralwidget = QtWidgets.QWidget(syncGraphMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mainOptionsBar = QtWidgets.QToolBox(self.centralwidget)
        self.mainOptionsBar.setMinimumSize(QtCore.QSize(300, 0))
        self.mainOptionsBar.setMaximumSize(QtCore.QSize(300, 16777215))
        self.mainOptionsBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainOptionsBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainOptionsBar.setObjectName("mainOptionsBar")
        self.leftMenuColumnWidget = QtWidgets.QWidget()
        self.leftMenuColumnWidget.setGeometry(QtCore.QRect(0, 0, 296, 700))
        self.leftMenuColumnWidget.setObjectName("leftMenuColumnWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.leftMenuColumnWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.checkFilesButton = QtWidgets.QPushButton(self.leftMenuColumnWidget)
        self.checkFilesButton.setMinimumSize(QtCore.QSize(0, 40))
        self.checkFilesButton.setObjectName("checkFilesButton")
        self.verticalLayout.addWidget(self.checkFilesButton)
        self.inFilesListView = QtWidgets.QListView(self.leftMenuColumnWidget)
        self.inFilesListView.setToolTipDuration(0)
        self.inFilesListView.setEditTriggers(QtWidgets.QAbstractItemView.CurrentChanged|QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.EditKeyPressed)
        self.inFilesListView.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.inFilesListView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.inFilesListView.setObjectName("inFilesListView")
        self.verticalLayout.addWidget(self.inFilesListView)
        self.inFilesToolButtonsLayout = QtWidgets.QHBoxLayout()
        self.inFilesToolButtonsLayout.setObjectName("inFilesToolButtonsLayout")
        self.addInFileButton = QtWidgets.QToolButton(self.leftMenuColumnWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.addInFileButton.sizePolicy().hasHeightForWidth())
        self.addInFileButton.setSizePolicy(sizePolicy)
        self.addInFileButton.setMinimumSize(QtCore.QSize(0, 30))
        self.addInFileButton.setObjectName("addInFileButton")
        self.inFilesToolButtonsLayout.addWidget(self.addInFileButton)
        self.removeInFileButton = QtWidgets.QToolButton(self.leftMenuColumnWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.removeInFileButton.sizePolicy().hasHeightForWidth())
        self.removeInFileButton.setSizePolicy(sizePolicy)
        self.removeInFileButton.setMinimumSize(QtCore.QSize(0, 30))
        self.removeInFileButton.setObjectName("removeInFileButton")
        self.inFilesToolButtonsLayout.addWidget(self.removeInFileButton)
        self.verticalLayout.addLayout(self.inFilesToolButtonsLayout)
        self.tableWidget = QtWidgets.QTableWidget(self.leftMenuColumnWidget)
        self.tableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.verticalLayout.addWidget(self.tableWidget)
        self.pushButton = QtWidgets.QPushButton(self.leftMenuColumnWidget)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.mainOptionsBar.addItem(self.leftMenuColumnWidget, "")
        self.additionalLeftBarPage = QtWidgets.QWidget()
        self.additionalLeftBarPage.setGeometry(QtCore.QRect(0, 0, 296, 700))
        self.additionalLeftBarPage.setObjectName("additionalLeftBarPage")
        self.mainOptionsBar.addItem(self.additionalLeftBarPage, "")
        self.horizontalLayout.addWidget(self.mainOptionsBar)
        self.chooseItemBar = QtWidgets.QFrame(self.centralwidget)
        self.chooseItemBar.setMaximumSize(QtCore.QSize(150, 16777215))
        self.chooseItemBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.chooseItemBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.chooseItemBar.setObjectName("chooseItemBar")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.chooseItemBar)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.loadButton = QtWidgets.QPushButton(self.chooseItemBar)
        self.loadButton.setMinimumSize(QtCore.QSize(0, 40))
        self.loadButton.setObjectName("loadButton")
        self.verticalLayout_2.addWidget(self.loadButton)
        self.listView = QtWidgets.QListView(self.chooseItemBar)
        self.listView.setObjectName("listView")
        self.verticalLayout_2.addWidget(self.listView)
        self.horizontalLayout.addWidget(self.chooseItemBar)
        self.tabView = QtWidgets.QTabWidget(self.centralwidget)
        self.tabView.setObjectName("tabView")
        self.graphTabPage = QtWidgets.QWidget()
        self.graphTabPage.setObjectName("graphTabPage")
        self.gridLayout = QtWidgets.QGridLayout(self.graphTabPage)
        self.gridLayout.setObjectName("gridLayout")
        self.contentGraphics = QtWidgets.QWidget(self.graphTabPage)
        self.contentGraphics.setObjectName("contentGraphics")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.contentGraphics)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.graphFrame = QtWidgets.QFrame(self.contentGraphics)
        self.graphFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.graphFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.graphFrame.setObjectName("graphFrame")
        self.horizontalLayout_4.addWidget(self.graphFrame)
        self.gridLayout.addWidget(self.contentGraphics, 0, 0, 1, 1)
        self.graphOptionsGroupBox = QtWidgets.QGroupBox(self.graphTabPage)
        self.graphOptionsGroupBox.setMaximumSize(QtCore.QSize(16777215, 300))
        self.graphOptionsGroupBox.setObjectName("graphOptionsGroupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.graphOptionsGroupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.groupBox_3 = QtWidgets.QGroupBox(self.graphOptionsGroupBox)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 3, 0, 1, 1)
        self.LEdit = QtWidgets.QSpinBox(self.groupBox_3)
        self.LEdit.setMinimumSize(QtCore.QSize(100, 28))
        self.LEdit.setMinimum(1)
        self.LEdit.setMaximum(65535)
        self.LEdit.setProperty("value", 2000)
        self.LEdit.setObjectName("LEdit")
        self.gridLayout_2.addWidget(self.LEdit, 3, 1, 1, 1)
        self.tstrtEdit = QtWidgets.QSpinBox(self.groupBox_3)
        self.tstrtEdit.setMinimumSize(QtCore.QSize(100, 28))
        self.tstrtEdit.setMaximum(65535)
        self.tstrtEdit.setObjectName("tstrtEdit")
        self.gridLayout_2.addWidget(self.tstrtEdit, 0, 1, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_3, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.graphOptionsGroupBox)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.Fpass2Edit = QtWidgets.QSpinBox(self.groupBox_2)
        self.Fpass2Edit.setMinimumSize(QtCore.QSize(100, 28))
        self.Fpass2Edit.setProperty("value", 35)
        self.Fpass2Edit.setObjectName("Fpass2Edit")
        self.gridLayout_3.addWidget(self.Fpass2Edit, 2, 1, 1, 1)
        self.Fpass1Edit = QtWidgets.QSpinBox(self.groupBox_2)
        self.Fpass1Edit.setMinimumSize(QtCore.QSize(100, 28))
        self.Fpass1Edit.setProperty("value", 20)
        self.Fpass1Edit.setObjectName("Fpass1Edit")
        self.gridLayout_3.addWidget(self.Fpass1Edit, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_2, 0, 1, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.graphOptionsGroupBox)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout_3.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout_3.addWidget(self.radioButton_2)
        self.gridLayout_6.addWidget(self.groupBox, 1, 0, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.graphOptionsGroupBox)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.levEdit = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.levEdit.setMinimumSize(QtCore.QSize(100, 28))
        self.levEdit.setProperty("value", 0.15)
        self.levEdit.setObjectName("levEdit")
        self.gridLayout_4.addWidget(self.levEdit, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_4)
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_4)
        self.label_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 1, 0, 1, 1)
        self.ksenEdit = QtWidgets.QSpinBox(self.groupBox_4)
        self.ksenEdit.setMinimumSize(QtCore.QSize(100, 28))
        self.ksenEdit.setProperty("value", 91)
        self.ksenEdit.setObjectName("ksenEdit")
        self.gridLayout_4.addWidget(self.ksenEdit, 1, 1, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_4, 1, 1, 1, 1)
        self.shiftsGroupBox = QtWidgets.QGroupBox(self.graphOptionsGroupBox)
        self.shiftsGroupBox.setObjectName("shiftsGroupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.shiftsGroupBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout_6.addWidget(self.shiftsGroupBox, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.graphOptionsGroupBox, 1, 0, 1, 1)
        self.tabView.addTab(self.graphTabPage, "")
        self.tableTabPage = QtWidgets.QWidget()
        self.tableTabPage.setObjectName("tableTabPage")
        self.tabView.addTab(self.tableTabPage, "")
        self.horizontalLayout.addWidget(self.tabView)
        syncGraphMainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(syncGraphMainWindow)
        self.tabView.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(syncGraphMainWindow)

    def retranslateUi(self, syncGraphMainWindow):
        _translate = QtCore.QCoreApplication.translate
        syncGraphMainWindow.setWindowTitle(_translate("syncGraphMainWindow", "Graphics synchronization"))
        self.checkFilesButton.setText(_translate("syncGraphMainWindow", "Check/Uncheck selected"))
        self.addInFileButton.setText(_translate("syncGraphMainWindow", "+"))
        self.removeInFileButton.setText(_translate("syncGraphMainWindow", "-"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("syncGraphMainWindow", "Property"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("syncGraphMainWindow", "Value"))
        self.pushButton.setText(_translate("syncGraphMainWindow", "Export"))
        self.mainOptionsBar.setItemText(self.mainOptionsBar.indexOf(self.leftMenuColumnWidget), _translate("syncGraphMainWindow", "Options"))
        self.mainOptionsBar.setItemText(self.mainOptionsBar.indexOf(self.additionalLeftBarPage), _translate("syncGraphMainWindow", "Additional"))
        self.loadButton.setText(_translate("syncGraphMainWindow", "Load"))
        self.groupBox_3.setTitle(_translate("syncGraphMainWindow", "Window"))
        self.label.setText(_translate("syncGraphMainWindow", "tstrt (start)"))
        self.label_2.setText(_translate("syncGraphMainWindow", "L ( lenght)"))
        self.groupBox_2.setTitle(_translate("syncGraphMainWindow", "Frequency"))
        self.label_4.setText(_translate("syncGraphMainWindow", "Fpass1 (start)"))
        self.label_5.setText(_translate("syncGraphMainWindow", "Fpass2 (end)"))
        self.groupBox.setTitle(_translate("syncGraphMainWindow", "Mode (compiling mode)"))
        self.radioButton.setText(_translate("syncGraphMainWindow", "debugging"))
        self.radioButton_2.setText(_translate("syncGraphMainWindow", "uploading"))
        self.groupBox_4.setTitle(_translate("syncGraphMainWindow", "Other"))
        self.label_7.setText(_translate("syncGraphMainWindow", "lev (incoming signal noise level)"))
        self.label_8.setText(_translate("syncGraphMainWindow", "ksen (meter sensor number)"))
        self.shiftsGroupBox.setTitle(_translate("syncGraphMainWindow", "dT (each file shifts)"))
        self.tabView.setTabText(self.tabView.indexOf(self.graphTabPage), _translate("syncGraphMainWindow", "Graph"))
        self.tabView.setTabText(self.tabView.indexOf(self.tableTabPage), _translate("syncGraphMainWindow", "Table"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    syncGraphMainWindow = QtWidgets.QMainWindow()
    ui = Ui_syncGraphMainWindow()
    ui.setupUi(syncGraphMainWindow)
    syncGraphMainWindow.show()
    sys.exit(app.exec_())
