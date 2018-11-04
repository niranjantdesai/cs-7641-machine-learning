from sklearn import datasets
from sklearn import svm


digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
y_pred = clf.predict(digits.data[-1:])
print(y_pred)