from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import utils
from data import Data
import sklearn.model_selection as cv


def main():
    file_data, file_classes = utils.load_data()
    data = Data(file_data, file_classes, 1,  0.7, 9)
    kFold = cv.KFold( n_splits=10, random_state=7, shuffle=True)
    lda = LDA()
    gnb = GaussianNB()

    print("Accuracy of methods:")
    calc_accuracy(data.data, data.classes, kFold, lda, gnb)

    print("Logarithmic Loss Results:")
    calc_loss(data.data, data.classes, kFold, lda, gnb)

    print("Area Under ROC Curve Results: ")
    #calc_curve(data.data, data.classes, kFold, lda, gnb)

    print("Confusion Matrixes:")
    calc_matrix(data.train_data, data.test_data, data.train_classes, data.test_classes, lda, gnb)

# Accuracy
def calc_accuracy(data, classes, kFold, lda, gnb):
    result = cv.cross_val_score(lda, data, classes, cv=kFold, scoring='accuracy')
    print(" LDA:")
    print(" - mean: %0.5f" % result.mean())
    print(" - standart deviation: %0.5f" % result.std())

    result = cv.cross_val_score(gnb, data, classes, cv=kFold, scoring='accuracy')
    print(" Gaussian:")
    print(" - mean: %0.5f" % result.mean())
    print(" - standart deviation: %0.5f" % result.std())


# Logarithmic Loss
def calc_loss(data, classes, kFold, lda, gnb):
    result = cv.cross_validate(lda, data, classes, cv=kFold, scoring='neg_log_loss',  return_train_score=False)
    result = result['test_neg_log_loss']
    print(" LDA:")
    print(" - mean: %0.5f" % result.mean())
    print(" - standart deviation: %0.5f" % result.std())
    result = cv.cross_val_score(gnb, data, classes, cv=kFold, scoring='neg_log_loss')
    print(" Gaussian:")
    print(" - mean: %0.5f" % result.mean())
    print(" - standart deviation: %0.5f" % result.std())


# Area Under ROC Curve
def calc_curve(ds_attr, ds_class, kFold, lda, gnb):
    result = cv.cross_val_score(lda, ds_attr, ds_class, cv=kFold, scoring='roc_auc')
    print(" LDA:")
    print(" - mean: %0.5f" % result.mean())
    print(" - standart deviation: %0.5f" % result.std())
    result = cv.cross_val_score(gnb, ds_attr, ds_class, cv=kFold, scoring='roc_auc')
    print(" Gaussian: %0.5f (%0.5f)" % (result.mean(), result.std() ))
    print(" - mean: %0.5f" % result.mean())
    print(" - standart deviation: %0.5f" % result.std())


# Confusion Matrix
def calc_matrix(X_train, X_test, Y_train, Y_test, lda, gnb):
    gnb.fit(X_train, Y_train)
    gnb_predicted = gnb.predict(X_test)
    gnb_matrix = confusion_matrix(Y_test, gnb_predicted)
    print(" - GaussianNB:")
    print(gnb_matrix)
    lda.fit(X_train, Y_train)
    lda_predicted = lda.predict(X_test)
    lda_matrix = confusion_matrix(Y_test, lda_predicted)
    print(" - LDA:")
    print(lda_matrix)

    # Classification Report
    print("Classification Reports:")
    lda_r = classification_report(Y_test, lda_predicted)
    print(' - LDA:')
    print(lda_r)
    gaus_r = classification_report(Y_test, gnb_predicted)
    print(" - GaussianNB:")
    print(gaus_r)


main()
