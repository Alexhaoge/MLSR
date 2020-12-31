import sys
# import os
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
from demo.page import *
from demo.pop import *
from MLSR.data import DataSet
from joblib import load


class childWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child = Ui_Dialog()
        self.child.setupUi(self)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.model_path = None
        self.model = None

    def model_predict(self):
        if self.model is None:
            # self.resultArea.setText('where is the model?')
            try:
                childWindow().exec_()
                self.model_path = self.open_model()
                self.model = load(self.model_path)
            except Exception as e:
                self.model = None
                return
        self.resultArea.setText('f')

    def open_model(self):
        dig = QFileDialog()
        dig.setFileMode(QFileDialog.AnyFile)
        dig.setFilter(QDir.Files)
        dig.exec_()
        print(dig.selectedFiles()[0])
        return dig.selectedFiles()[0]

    def load_from_status_bar(self):
        try:
            self.model_path = self.open_model()
            self.model = load(self.model_path)
        except Exception as e:
            self.model = None

    def open_github(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://github.com/Alexhaoge/MLSR'))


if __name__ == '__main__':
    # os.environ["LOG4CPLUS_LOGLOG_QUIETMODE"] = 'true'
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
