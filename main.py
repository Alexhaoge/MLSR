from MLSR.data import DataSet
from MLSR.primary import do_decision_tree

x = DataSet('data/rand_select_400_avg.csv')
x.generate_feature()
y = DataSet('data/not_selected_avg.csv')
y.generate_feature()
fake = DataSet.data_augment()
z = DataSet.static_merge(x, y)
zz = DataSet.static_merge(z, fake)
print(zz.label.value_counts())
do_decision_tree(zz, 'log/dTree')
