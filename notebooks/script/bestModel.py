import pandas as pd
from sklearn.utils import all_estimators
from .evaluation import eval
from sklearn.model_selection import train_test_split

def bestModel(X, y):
    classifier = []
    best_classifier = None
    best_accuracy = 0.0

    for class_ in all_estimators(type_filter='classifier'):
        try:
            classifier_instance = class_[1]()
            classifier.append(classifier_instance)
        except:
            pass

    for model in classifier:
        try:
            model, accuracy = eval(model, X, y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = model
        except:
            pass

    return best_classifier
