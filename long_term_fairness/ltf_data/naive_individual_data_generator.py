"""Generate Data"""

import numpy as np
from .data_generator_base import DataBaseClass


class DataGenerator(DataBaseClass):
    """"""

    def sample(self, X, _y, y_hat):
        """

        Args:
            X: 3D array, the features at each time step t [t, X]
            _y: ignored, the true labels
            y_hat: 2D array, predictions for each individual i over past t time steps [t, y_hat]

        Returns:

        """

        if y_hat is None:
            return self._generate_initial_data()

        samples = []
        labels = []

        for i, _ in enumerate(X[-1]):
            s = self._get_probability(y_hat, i)  # + self._OFFSET
            r = np.random.uniform()

            if s > r:
                samples.append(np.random.multivariate_normal(self._mean_pos, self._cov_pos, 1).ravel())
                labels.append(self._pos_label)
            else:
                samples.append(np.random.multivariate_normal(self._mean_neg, self._cov_neg, 1).ravel())
                labels.append(self._neg_label)

        samples = np.asarray(samples)
        labels = np.asarray(labels)

        return samples, self._sens_attrs, labels

    def _get_probability(self, y_hat, i):
        """"""
        s = 0
        for t in range(1, self._degree + 1):
            if len(y_hat) >= t:
                s += y_hat[-t][i]
        return s / np.amin((len(y_hat), self._degree))
