import sys
from traceback import print_exc
from pandas import Series, DataFrame
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
from demo.page import *
from demo.pop import *
from MLSR.data import DataSet
from joblib import load
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


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
                print_exc(e)
                self.model = None
                return
        s = Series(dtype=object)
        s['f0'] = self.isCardProverty.isChecked()
        s['f1'] = self.isLowest.isChecked()
        s['f2'] = self.isFiveGuarantee.isChecked()
        s['f3'] = self.isOrphan.isChecked()
        s['f4'] = self.isMartyrsFamily.isChecked()
        s['f5'] = self.isBussiness.isChecked()
        s['f6'] = self.isFarm.isChecked()
        s['f7'] = self.isRetire.isChecked()
        s['f8'] = self.noIncome.isChecked()
        s['f9'] = self.isWork.isChecked()
        s['f10'] = self.bothUnemployed.isChecked()
        s['f11'] = self.eitherUnemployed.isChecked()
        s['f12'] = self.income.value() / self.household.value()
        s['f13'] = self.numUniv.value()
        s['f14'] = self.numHigh.value()
        s['f15'] = self.numPrim.value()
        s['f16'] = self.grandParentDisease.isChecked()
        s['f17'] = self.parentDivorce.isChecked()
        s['f18'] = self.oneParentNormalDisease.isChecked()
        s['f19'] = self.bothParentNormalDisease.isChecked()
        s['f20'] = self.siblingDisease.isChecked()
        s['f21'] = self.oneParentSeriousDisease.isChecked()
        s['f22'] = self.bothParentSeriousDisease.isChecked()
        s['f23'] = self.parentPassAway.isChecked()
        s['f24'] = self.naturalAccident.isChecked()
        s['f29'] = self.household.value()
        s['f30'] = self.yesLoan.isChecked()
        s['f31'] = self.isRuralResident.isChecked()
        try:
            ss = Series(dtype=object)
            ss['0'] = self.ethnic.text()
            s['f28'] = DataSet.do_ethnic_group(ss)['0']
            ss['0'] = self.scholarshipText.toPlainText().replace('\n', '')
            d = DataSet.do_scholarship(ss)
            s['f25'] = d['助学金个数']['0']
            s['f26'] = d['助学金金额']['0']
            s['f27'] = d['国助类型']['0']
            d = DataFrame(s.to_dict(), index=[0])
            ans = self.model.predict(d)[0]
            ans_type = ['非常困难', '一般困难', '可能为非困难生']
            self.resultArea.setText(ans_type[ans])
        except Exception as e:
            self.resultArea.setText('输入有误或模型导入错误\n请检查输入或重新导入模型')
            print_exc(e)

    def open_model(self):
        dig = QFileDialog()
        dig.setFileMode(QFileDialog.AnyFile)
        dig.setFilter(QDir.Files)
        dig.exec_()
        file_path = dig.selectedFiles()[0]
        print(file_path)
        return file_path

    def load_from_status_bar(self):
        try:
            self.model_path = self.open_model()
            self.model = load(self.model_path)
        except Exception as e:
            print_exc(e)
            self.model = None

    def open_github(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://github.com/Alexhaoge/MLSR'))


if __name__ == '__main__':
    # os.environ["LOG4CPLUS_LOGLOG_QUIETMODE"] = 'true'
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
