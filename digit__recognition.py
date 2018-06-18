from sklearn import tree
from sklearn import datasets

clf=tree.DecisionTreeClassifier()

datas=datasets.load_digits()
x,y=datas.data[:-1],datas.target[:-1]
clf.fit(x,y)
print(clf.predict(x[[-1]]))
