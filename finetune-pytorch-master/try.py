from   sklearn.model_selection  import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
data = load_iris()
train = data.data
test = data.target
print(train)
