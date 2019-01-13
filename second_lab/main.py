from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import utils
from data import Data


def calculate_accuracy(data):
    dtc = DecisionTreeClassifier()
    dtc.fit(data.train_data, data.train_classes)
    dtc_accuracy = accuracy_score(data.test_classes, dtc.predict(data.test_data))

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(data.train_data, data.train_classes)
    rfc_accuracy = accuracy_score(data.test_classes, rfc.predict(data.test_data))
    return dtc_accuracy, rfc_accuracy


def main():
    file_data, file_classes = utils.load_data()
    set_test_coefficients = [0.6, 0.7, 0.8, 0.9]

    for c in set_test_coefficients:
        dtc_accuracy, rfc_accuracy = calculate_accuracy(Data(file_data, file_classes, 1, c))
        text = '{}, {}, {}\n'.format(c * 100, dtc_accuracy * 100, rfc_accuracy * 100)
        print(text)


main()
