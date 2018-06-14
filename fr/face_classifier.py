import datetime
import pickle

import numpy as np
from sklearn import svm


class SVMClassifier:
    def __init__(self, probability=False):
        self.probability = probability
        if not probability:
            self.classifier = svm.LinearSVC(multi_class='ovr', random_state=0)

        else:
            self.classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
        self.class_names = []

    def train(self, inputs, labels, class_names, classifier_file=None, save_file=True):
        self.classifier.fit(np.array(inputs), np.array(labels))
        self.class_names = class_names
        if save_file:
            if classifier_file is None:
                classifier_file = 'classifier_{}.pkl'.format(datetime.datetime.now())
            with open(classifier_file, 'wb') as outfile:
                pickle.dump((self.classifier, class_names), outfile)

    def load(self, classifier_file):
        with open(classifier_file, 'rb') as outfile:
            self.classifier, self.class_names = pickle.load(outfile)

    def predict(self, x):
        if not self.probability:
            predictions = self.classifier.predict(np.array(x))

            return [self.class_names[i] for i in predictions]
        else:
            predictions = self.classifier.predict_proba(np.array(x))
            best_indices = np.argmax(predictions, axis=1)
            best_probabilities = predictions[np.arange(len(best_indices)), best_indices]

            results = []
            for i in range(len(best_indices)):
                results.append((self.class_names[best_indices[i]], best_probabilities[i]))

            return results
