import datetime
import pickle

import numpy as np
from sklearn import svm


class SVMClassifier:
    def __init__(self):
        self.classifier = svm.SVC(probability=True)
        self.class_names = []

    def train(self, inputs, labels, class_names, classifier_file=None):
        self.classifier.fit(inputs, labels)
        self.class_names = class_names
        if classifier_file is None:
            classifier_file = 'classifier_{}.pkl'.format(datetime.datetime.now())
        with open(classifier_file, 'wb') as outfile:
            pickle.dump((self.classifier, class_names), outfile)

    def load(self, classifier_file):
        with open(classifier_file, 'rb') as outfile:
            self.classifier, self.class_names = pickle.load(outfile)

    def predict(self, x):
        predictions = self.classifier.predict_proba(x)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        results = []
        for i in range(len(best_class_indices)):
            results.append((self.class_names[best_class_indices[i]], best_class_probabilities[i]))

        return results
