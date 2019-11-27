import numpy as np


class DataGenerator:
    """"""

    def __init__(self, mean_1=[10, 7], cov_1=[[1.2, 0], [0, 1.3]],
                 mean_2=[0, 1], cov_2=[[1, 0], [0, 1.2]], degree=3, discrimination_factor=.7,
                 num_positive_samples=500, num_negative_samples=500, neg_label=0, pos_label=1):
        """"""
        assert np.shape(mean_1) == np.shape(mean_2)
        assert np.shape(cov_1) == np.shape(cov_2)

        self._mean_pos = mean_1
        self._mean_neg = mean_2
        self._cov_pos = cov_1
        self._cov_neg = cov_2

        self._num_pos = num_positive_samples
        self._num_neg = num_negative_samples

        self._sens_attrs = np.random.randint(0, 2, size=num_negative_samples+num_positive_samples)  # TODO:

        self._degree = degree
        self._neg_label = neg_label
        self._pos_label = pos_label

    def generate_data(self, X, _y, y_hat):
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
            s = self._get_sum(y_hat, i)
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

    def _generate_initial_data(self, ):
        """"""
        pos_samples = np.random.multivariate_normal(self._mean_pos,
                                                    self._cov_pos,
                                                    self._num_pos)

        neg_samples = np.random.multivariate_normal(self._mean_neg,
                                                    self._cov_neg,
                                                    self._num_neg)

        samples = np.vstack((pos_samples, neg_samples))
        labels = np.hstack((np.ones(self._num_pos) * self._pos_label,
                            np.ones(self._num_neg) * self._neg_label))

        return samples, self._sens_attrs, labels

    def _get_sum(self, y_hat, i):
        """"""
        s = 0
        for t in range(1, self._degree + 1):

            if len(y_hat) >= t:
                s += y_hat[-t][i]
        return s / len(y_hat)
