from MLSR.data import DataSet
from MLSR.primary import do_decision_tree

x = DataSet('data/rand_select_400_avg.csv')
x.generate_feature()
y = DataSet('data/not_selected_avg.csv')
y.generate_feature()
z = DataSet.static_merge(x, y)
do_decision_tree(z, 'log/dTree')