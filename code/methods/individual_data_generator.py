import numpy as np


class DataGenerator:
    """"""

    def __init__(self, mean_1=[10, 7], cov_1=[[1.2, 0], [0, 1.3]], mean_2=[0, 1],
                 cov_2=[[1, 0], [0, 1.2]], degree=3, discrimination_factor=.7, num_positive_samples=500,
                 num_negative_samples=500, step_size=.03, local_var=[[.02, 0], [0, .02]], neg_label=0, pos_label=1):
        """"""
        assert np.shape(mean_1) == np.shape(mean_2)
        assert np.shape(cov_1) == np.shape(cov_2)

        self._mean_pos = mean_1
        self._mean_neg = mean_2
        self._cov_pos = cov_1
        self._cov_neg = cov_2

        self._num_pos = num_positive_samples
        self._num_neg = num_negative_samples
        self._step_size = step_size

        self._sens_attrs = np.random.randint(0, 2, size=num_negative_samples+num_positive_samples)  # TODO:

        self._local_variance = local_var

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

        new_X = []
        labels = []
        for i, x_i in enumerate(X[-1]):
            y_hat_i = [y_hat[t][i] for t in range(len(y_hat))]
            new_x_i = self._sample_point(x_i, y_hat_i)
            new_X.append(new_x_i)
            labels.append(self._get_label(new_x_i))

        samples = np.asarray(new_X)
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

        # TODO: shuffle
        samples = np.vstack((pos_samples, neg_samples))
        labels = np.hstack((np.ones(self._num_pos) * self._pos_label,
                            np.ones(self._num_neg) * self._neg_label))

        return samples, self._sens_attrs, labels

    def _get_label(self, x_i):
        """

        Args:
             x_i: vector, the current features of individual i

        Returns:

        """
        distance_pos_mean = np.linalg.norm((x_i - self._mean_pos))
        distance_neg_mean = np.linalg.norm((x_i - self._mean_neg))

        if distance_pos_mean < distance_neg_mean:
            return self._pos_label
        return self._neg_label

    def _sample_point(self, x_i, y_hat_i):
        """

        Args:
            x_i: vector, the current features of individual i
            y_hat_i: vector, all past predictions for individual i

        Returns:

        """
        cov = self._cov_function()
        mean = self._mean_function(x_i, y_hat_i)
        new_point = np.random.multivariate_normal(mean, cov, 1)
        return new_point.ravel()

    def _cov_function(self):
        """"""
        return self._local_variance

    def _mean_function(self, x_i, y_hat_i):
        """x + sum(y_hat * alpha * v)"""
        s = 0
        for t in range(1, self._degree + 1):
            if len(y_hat_i) >= t:
                s += self._step_size * self._direction_vector(x_i, y_hat_i[-t])

        return x_i + s

    def _direction_vector(self, x_i, y_hat_i_t):
        """

        Args:
            x_i: vector,  the features of individual i
            y_hat_i_t: scalar, the value of y_hat for individual i at time step t

        Returns:

        """
        if y_hat_i_t:
            return self._mean_pos - x_i
        else:
            return self._mean_neg - x_i
