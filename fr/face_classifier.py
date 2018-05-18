import datetime
import pickle

import numpy as np
from sklearn import svm


class SVMClassifier:
    def __init__(self):
        self.classifier = svm.LinearSVC(multi_class='crammer_singer', random_state=0)
        self.class_names = []

    def train(self, inputs, labels, class_names, classifier_file=None):
        self.classifier.fit(np.array(inputs), np.array(labels))
        self.class_names = class_names
        if classifier_file is None:
            classifier_file = 'classifier_{}.pkl'.format(datetime.datetime.now())
        with open(classifier_file, 'wb') as outfile:
            pickle.dump((self.classifier, class_names), outfile)

    def load(self, classifier_file):
        with open(classifier_file, 'rb') as outfile:
            self.classifier, self.class_names = pickle.load(outfile)

    def predict(self, x):
        predictions = self.classifier.predict(np.array(x))

        return [self.class_names[i] for i in predictions]