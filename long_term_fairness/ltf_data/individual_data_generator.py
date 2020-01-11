""""""
import numpy as np
from .data_generator_base import DataBaseClass


class DataGenerator(DataBaseClass):
    """"""

    def __init__(self, mean_pos=[10, 7], cov_pos=[[1.2, 0], [0, 1.3]],
                 mean_neg=[0, 1], cov_neg=[[1, 0], [0, 1.2]], degree=3, discrimination_factor=.7,
                 num_positive_label=500, num_negative_label=500, neg_label=0, pos_label=1,
                 local_variance=[[0.1, 0], [0, 0.1]], step_size=0.1, neg_class=0, pos_class=1):
        """"""
        super().__init__(mean_pos=mean_pos, cov_pos=cov_pos, mean_neg=mean_neg, cov_neg=cov_neg,
                         degree=degree, discrimination_factor=discrimination_factor,
                         num_positive_label=num_positive_label,
                         num_negative_label=num_negative_label, neg_label=neg_label, pos_label=pos_label,
                         neg_class=neg_class, pos_class=pos_class)

        self._local_variance = local_variance
        self._step_size = step_size

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

        new_X = []
        for i, x_i in enumerate(X[-1]):
            y_hat_i = [y_hat[t][i] for t in range(len(y_hat))]
            new_x_i = self._sample_point(x_i, y_hat_i)
            new_X.append(new_x_i)

        samples = np.asarray(new_X)
        labels = self.get_label(samples)

        return samples, self._sens_attrs, labels

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
                s += y_hat_i[-t]  # self._step_size * self._direction_vector(x_i, y_hat_i[-t])

        s = self._step_size * self._direction_vector(x_i, y_hat_i[-1]) * s
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
