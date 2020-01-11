"""Base class for all ltf_data generators"""

from abc import ABC, abstractmethod
import numpy as np


class DataBaseClass(ABC):
    _OFFSET = .2

    def __init__(self, mean_pos=[10, 7], cov_pos=[[1.2, 0], [0, 1.3]],
                 mean_neg=[0, 1], cov_neg=[[1, 0], [0, 1.2]], degree=3, discrimination_factor=.7,
                 num_positive_label=500, num_negative_label=500, neg_label=0, pos_label=1, neg_class=0, pos_class=1):
        """"""
        assert np.shape(mean_pos) == np.shape(mean_neg)
        assert np.shape(cov_pos) == np.shape(cov_neg)

        self._mean_pos = mean_pos
        self._mean_neg = mean_neg
        self._cov_pos = cov_pos
        self._cov_neg = cov_neg

        self._initial_num_pos_label = num_positive_label
        self._initial_num_neg_label = num_negative_label

        num_pos_class_pos_lbl = int(num_positive_label * discrimination_factor)
        num_neg_class_pos_lbl = num_positive_label - num_pos_class_pos_lbl

        num_neg_class_neg_lbl = int(num_negative_label * discrimination_factor)
        num_pos_class_neg_lbl = num_negative_label - num_neg_class_neg_lbl

        pos_cls_pos_lbl_samples = np.ones(num_pos_class_pos_lbl) * pos_class
        neg_cls_pos_lbl_samples = np.ones(num_neg_class_pos_lbl) * neg_class

        pos_cls_neg_lbl_samples = np.ones(num_pos_class_neg_lbl) * pos_class
        neg_cls_neg_lbl_samples = np.ones(num_neg_class_neg_lbl) * neg_class

        # the top num_positive_label items belong all have a positive label
        # (the ltf_data is not shuffled)
        self._sens_attrs = np.hstack((pos_cls_pos_lbl_samples,
                                      neg_cls_pos_lbl_samples,
                                      pos_cls_neg_lbl_samples,
                                      neg_cls_neg_lbl_samples))

        self._num_protected_individuals = num_neg_class_neg_lbl + num_neg_class_pos_lbl
        self._num_unprotected_individuals = num_pos_class_neg_lbl + num_pos_class_pos_lbl

        self._degree = degree
        self._neg_label = neg_label
        self._pos_label = pos_label

        self._neg_class = neg_class
        self._pos_class = pos_class

    def _generate_initial_data(self, ):
        """"""
        pos_samples = np.random.multivariate_normal(self._mean_pos,
                                                    self._cov_pos,
                                                    self._initial_num_pos_label)

        neg_samples = np.random.multivariate_normal(self._mean_neg,
                                                    self._cov_neg,
                                                    self._initial_num_neg_label)

        samples = np.vstack((pos_samples, neg_samples))
        labels = np.hstack((np.ones(self._initial_num_pos_label) * self._pos_label,
                            np.ones(self._initial_num_neg_label) * self._neg_label))

        return samples, self._sens_attrs, labels

    def get_label(self, X, _X_sens=None):
        """

        Args:
             X: 2D, the current features

        Returns:

        """
        distance_pos_mean = np.linalg.norm((X - self._mean_pos), axis=1)
        distance_neg_mean = np.linalg.norm((X - self._mean_neg), axis=1)

        labels = distance_pos_mean < distance_neg_mean

        int_labels = np.ones_like(labels) * self._neg_label
        int_labels[labels] = self._pos_label

        return int_labels

    @abstractmethod
    def sample(self, X, _y, y_hat):
        pass
