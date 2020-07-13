"""Generate data where whole group benefits from positive decisions.

Methods are nealy identical to the individual data generator. Some should have been
merged into the base class.

TODO:
    * stop when individuals crossed decision boundary (add negative momentum or similar)
    * move some methods to the base class
"""
import numpy as np
from .data_generator_base import DataGeneratorBase


class GroupDataGenerator(DataGeneratorBase):
    """"""

    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)

    def sample(self, X, _y, y_hat):
        """Sample new points.

        Args:
            X (array_like): 3D, the old features at each time step t [t, X].
            _y (array_like): Ignored, the true labels for all time steps t [t, y].
            y_hat (array_like): 2D, previous predictions [t, y_hat].

        Returns:
            X_t, X_sens_t, y_t (numpy.ndarray, numpy.ndarray, numpy.ndarray): 2D, 1D, 1D.
        """

        if y_hat is None:
            return self._sample_initial_data()

        new_X = []
        for i, x_i in enumerate(X[-1]):
            group_mask = self._sens_attr[i] == self._sens_attr

            y_hat_g = [y_hat[t][group_mask] for t in range(len(y_hat))]

            new_x_i = self._sample_point(x_i, y_hat_g)
            new_X.append(new_x_i)

        samples = np.asarray(new_X)
        labels = self.get_label(samples)

        return samples, self._sens_attr, labels

    def _sample_point(self, x_i_t, y_hat_g):
        """ Sample one new point.

        Args:
            x_i_t (array_like): 1D, the current features of individual i.
            y_hat_g (array_like): 2D, previous predictions for group G influencing individual i.

        Returns:
            (numpy.ndarrary): The new point for individual i at time t+1.
        """
        cov = self._cov_function()
        mean = self._mean_function(x_i_t, y_hat_g)
        new_point = np.random.multivariate_normal(mean, cov, 1)
        return new_point.ravel()

    def _mean_function(self, x_i_t, y_hat_g):
        """Returns the new mean for individual i.

        The new mean is computed from the current point x_i_t plus some offset towards
        positive or negative cluster proportional to the sum of previous predictions for group G.

            mu =    x_i_t    +     step_size * direction_vector * sum(y_hat_g)

        Args:
            x_i_t (array_like): 1D, the features of individual i at time t.
            y_hat_g (array_like): 2D, previous predictions for the group G influencing individual i.

        Returns:
            (array_like): The new mean.
        """
        offset = 0

        for t in range(1, self.env_params.degree + 1):
            if len(y_hat_g) >= t:
                offset += np.sum(y_hat_g[-t]) / len(y_hat_g[-t])

        offset = self.env_params.step_size * self._direction_vector(x_i_t, offset) * offset

        return x_i_t + offset
