from flask import Flask, request, render_template
from MLSR.data import DataSet
from joblib import load
from traceback import print_exc
from pandas import Series, DataFrame

app = Flask(__name__)

model_select = {
    '决策树': [{'name': '决策树模型1', 'value': 0}],
    '随机森林': [
        {'name': '随机森林小模型', 'value': 1},
        {'name': '随机森林大模型', 'value': 2}
    ],
    '支持向量机': [{'name': '支持向量模型1', 'value': 3}],
    '逻辑回归': [{'name': '逻辑回归模型1', 'value':4}],
}
model_path = [
    '../log/dTree/best_model',
    '../log/rf/best_model',
    '../log/rf/best_model_1',
    '../log/svm/best_model_1',
    '../log/lr/best_model'
]

required_fieldset = [
    'isCardProverty', 'isLowest', 'isFiveGuarantee', 'isMartyrsFamily',
    'isParentsEmployed', 'isWork', 'isFarm', 'isBussiness', 'isRetire', 'noIncome',
    'income', 'household', 'numUniv', 'numHigh', 'numPrim',
    'grandParentDisease', 'parentDivorce', 'oneParentNormalDisease',
    'bothParentNormalDisease', 'oneParentSeriousDisease',
    'bothParentSeriousDisease', 'siblingDisease', 'parentPassAway', 
    'naturalAccident', 'isRuralResident', 'yesLoan',
    'ethnic', 'scholarship', 'selectModel',
]

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', model_select = model_select)
    else:
        form = request.form
        for field in required_fieldset:
            if not field in form:
                return {'error': f'缺少字段{field}'}
        if form['household'] < form['numUniv'] + form['numHigh'] + form['numPrim']:
            return {'error': '家庭人数不应小于在受教育家庭成员人数'}
        try:
            return model_predict(form)
        except ValueError as ve:
            return {'error': ve.__str__()}
        except Exception as e:
            return {'error': 'Unknown error: ' + e.__str__()}


def model_predict(form):
#    return {'not': 0.2, 'medium': 0.5, 'severe': 0.3, 'message': 'suceed'}
    try:
        model = load(model_path[int(form['selectModel'])])
    except Exception as e:
        app.logger.error(e)
        raise ValueError('load model error')
    s = Series(dtype=object)
    s['f0'] = form['isCardProverty'] == '1'
    s['f1'] = form['isLowest'] == '1'
    s['f2'] = form['isFiveGuarantee'] == '1'
    s['f3'] = form['isOrphan'] == '1'
    s['f4'] = form['isMartyrsFamily'] == '1'
    s['f5'] = form['isBussiness'] == '1'
    s['f6'] = form['isFarm'] == '1'
    s['f7'] = form['isRetire'] == '1'
    s['f8'] = form['noIncome'] == '1'
    s['f9'] = form['isWork'] == '1'
    s['f10'] = form['isParentsEmployed'] == '1'
    s['f11'] = form['isParentsEmployed'] == '2'
    s['f12'] = float(form['income']) / int(form['household'])
    s['f13'] = int(form['numUniv'])
    s['f14'] = int(form['numHigh'])
    s['f15'] = int(form['numPrim'])
    s['f16'] = form['grandParentDisease'] == '1'
    s['f17'] = form['parentDivorce'] == '1'
    s['f18'] = form['oneParentNormalDisease'] == '1'
    s['f19'] = form['bothParentNormalDisease'] == '1'
    s['f20'] = form['siblingDisease'] == '1'
    s['f21'] = form['oneParentSeriousDisease'] == '1'
    s['f22'] = form['bothParentSeriousDisease'] == '1'
    s['f23'] = form['parentPassAway'] == '1'
    s['f24'] = form['naturalAccident'] == '1'
    s['f28'] = form['ethnic'] == '1'
    s['f29'] = int(form['household'])
    s['f30'] = form['yesLoan'] == '1'
    s['f31'] = form['isRuralResident'] == '1'
    try:
        ss = Series(dtype=object)
        ss['0'] = form['scholarship'].replace('\n', '')
        d = DataSet.do_scholarship(ss)
        s['f25'] = d['助学金个数']['0']
        s['f26'] = d['助学金金额']['0']
        s['f27'] = d['国助类型']['0']
        d = DataFrame(s.to_dict(), index=[0])
        ans = model.predict_proba(d)[0]
        return {'not': ans[2], 'medium': ans[1], 'severe': ans[0], 'message': 'success'}
    except Exception as e:
        app.logger.error(e)
        raise ValueError('输入有误或模型导入错误\n请检查输入或重新导入模型')

