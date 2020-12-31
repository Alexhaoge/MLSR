import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
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
        self.msg = childWindow()
        try:
            self.model = load('best_model')
        except Exception as e:
            self.model = None

    def model_predict(self):
        if self.model is None:
            # self.resultArea.setText('where is the model?')
            self.msg.show()
            return
        self.resultArea.setText('f')

    def open_github(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://github.com/Alexhaoge/MLSR'))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
