from MLSR.data import DataSet
from MLSR.plot import *

x = DataSet('data/rand_select_400_avg.csv')
x.generate_feature()
y = DataSet('data/not_selected_avg.csv')
y.generate_feature()
z = DataSet.static_merge(x, y)
#plot_tsne(z, 'log/tsne.png')
z = z.convert_to_ssl()
z0, z1 = z.split_by_weak_label()
# plot_tsne_ssl(z0, 'log/0_tsne.png', n_iter=300)
plot_tsne_ssl(z1, 'log/1_tsne.png', n_iter=500)
