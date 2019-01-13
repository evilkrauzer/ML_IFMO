from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from data import Data
from first_lab.KNearestNeighbors import KNearestNeighbors
from first_lab.NaiveBayes import NaiveBayes
import utils

file_data, file_classes = utils.load_data()
data = Data(file_data, file_classes, 1, 0.7)

# NB
gnb = GaussianNB()
nb_clf = gnb.fit(data.train_data, data.train_classes)
accuracy = accuracy_score(data.test_classes, gnb.predict(data.test_data))
print('Naive Bayes Accuracy: %.8f' % accuracy)

# my NB
bayes_native = NaiveBayes()
bayes_native.fit(data.train_data, data.train_classes)
bayes_native_accuracy = accuracy_score(data.test_classes, bayes_native.predict(data.test_data))
print('my naive Bayes Accuracy: %.8f' % bayes_native_accuracy)

# KNN
sklearn_kn_clf = KNeighborsClassifier(10)
sklearn_kn_clf.fit(data.train_data, data.train_classes)
sklearn_kn_clf_accuracy = accuracy_score(data.test_classes, sklearn_kn_clf.predict(data.test_data))
print('knn Accuracy: {:.8%}'.format(sklearn_kn_clf_accuracy))

# My KNN
my_kn_clf = KNearestNeighbors(10)
my_kn_clf.fit(data.train_data, data.train_classes)
my_kn_clf_accuracy = accuracy_score(data.test_classes, my_kn_clf.predict(data.test_data))
print('my knn Accuracy: {:.8%}'.format(my_kn_clf_accuracy))
