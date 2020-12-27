from MLSR.tsvm import TSVM
from sklearn.base import is_classifier

t = TSVM()
print(is_classifier(t))