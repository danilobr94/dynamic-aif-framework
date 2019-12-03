"""Estimate effect of random changes to fairness

Steps:
    1. change data set
    2. make predictions on new data set
    3. evaluate with given aif360

"""

import numpy as np
from aif360.datasets.structured_dataset import StructuredDataset
from .data.dataset_mutator import DataMutator


class CrossFairness:

    def __init__(self, dataset: StructuredDataset, clf, metric):
        """

        Args:
            dataset:
            clf:
            metric:
        """
        self.dataset = dataset
        self.clf = clf
        self.metric = metric
        self._evaluations = {}
        self.dataset_mutator = DataMutator(dataset)

    def shuffle_feature(self, feature):
        """"""
        dataset_new = self.dataset_mutator.shuffle_attribute(feature)
        self._make_evaluations(dataset_new, "shuffled feature " + str(feature))

    def randomize_feature(self, feature, random_generator=None):
        """"""
        dataset_new = self.dataset_mutator.randomize_attribute(feature, random_generator=random_generator)
        self._make_evaluations(dataset_new, "randomized feature " + str(feature))

    def remove_feature(self, feature):
        """Remove feature from data set and check outcome"""
        dataset_new = self.dataset.copy(deepcopy=True)
        dataset_new.features = np.delete(self.dataset.features, feature, 1)

        self._make_evaluations(dataset_new, "removed feature " + str(feature))

    def remove_group(self, feature, group):
        """Remove all individuals from group in feature from data set"""
        dataset_new = self.dataset.copy(deepcopy=True)
        mask = dataset_new.features[feature] == group
        dataset_new.features = np.delete(dataset_new.features, mask, 0)

    def substitute_group(self, feature, group, new_group):
        """"""
        dataset_new = self.dataset_mutator.substitute_group(feature, group, new_group)
        self._make_evaluations(dataset_new, "removed group " + str(group))

    def _make_evaluations(self, dataset: StructuredDataset, label):
        """"""
        train, test = dataset.split([0.7], shuffle=True)
        clf = self.clf.fit(train)
        predictions = clf.predict(test)

        self._evaluations[label] = self.metric(test, predictions)

    def plot_evaluation(self):
        """"""
        print(self._evaluations)
        print("TODO: make plot")

