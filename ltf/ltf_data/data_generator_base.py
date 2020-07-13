"""Base class for data generators.

All generators sample data from two gaussian blobs, a negative and a positive labeled one.
Data initialization is the same for all generators as-well.
Points are assigned the label of the closer cluster.
The protected feature is constant over time and sampled on initialization.
"""

from abc import ABC, abstractmethod
import numpy as np


class _EnvParams:
    """Environment Parameters."""

    def __init__(self, step_size, neg_label, pos_label, neg_class, pos_class,
                 mean_pos, mean_neg, cov_pos, cov_neg, local_variance, degree):
        """"""

        self.step_size = step_size

        self.neg_label = neg_label
        self.pos_label = pos_label
        self.neg_class = neg_class
        self.pos_class = pos_class

        self.mean_pos = np.asarray(mean_pos)
        self.mean_neg = np.asarray(mean_neg)
        self.cov_pos = np.asarray(cov_pos)
        self.cov_neg = np.asarray(cov_neg)

        self.local_variance = local_variance
        self.degree = degree

    def pos_dir(self, x_i_t):
        """Return direction vector for movement towards positive cluster.

        A random point is sampled, to avoid convergence to the mean value.
        """
        random_p = np.random.multivariate_normal(self.mean_pos,
                                                 self.cov_pos, 1).ravel()
        return random_p - x_i_t

    def neg_dir(self, x_i_t):
        """Return direction vector for movement towards negative cluster.

        A random point is sampled, to avoid convergence to the mean value.
        """
        random_p = np.random.multivariate_normal(self.mean_neg,
                                                 self.cov_neg, 1).ravel()
        return random_p - x_i_t


class DataGeneratorBase(ABC):
    """"""

    def __init__(self, mean_pos=(10, 7), cov_pos=((1.2, 0), (0, 1.3)),
                 mean_neg=(0, 1), cov_neg=((1, 0), (0, 1.2)), degree=3,
                 discrimination_factor=.7, num_positive_label=500, num_negative_label=500,
                 neg_label=0, pos_label=1, neg_class=0, pos_class=1,
                 local_variance=((0.1, 0), (0, 0.1)), step_size=0.1, ):
        """Initialize the data generator.

        Computes the number of individuals for each group and samples the protected attribute.

        Args:
            mean_pos (array_like): 1D, mean of the positive cluster of shape [D].
            cov_pos (array_like): 2D, covariance of the positive cluster of shape [D, D].
            mean_neg (array_like): 1D, mean of the negative cluster of shape [D].
            cov_neg (array_like): 2D, covariance of the negative cluster of shape [D, D].
            degree (int): The number of previous time steps considered for data generation.
            discrimination_factor (float): Proportion of protected attribute per label on init.
            num_positive_label (int): Number of individuals with positive label.
            num_negative_label (int): Number of individuals with negative label.
            pos_class (int): The positive class of the protected attribute.
            neg_class (int): The negative class of the protected attribute.
            pos_label (int): The positive label.
            neg_label (int): The negative label.
            local_variance (array_like): 2D, the variance to sample new points from.
            step_size (float): The fraction each individual moves towards the other cluster.
        """
        assert np.shape(mean_pos) == np.shape(mean_neg)
        assert np.shape(cov_pos) == np.shape(cov_neg)

        # Environment parameters.
        self.env_params = _EnvParams(step_size, neg_label, pos_label, neg_class, pos_class,
                                     mean_pos, mean_neg, cov_pos, cov_neg, local_variance, degree)

        self._initial_num_pos_label = num_positive_label
        self._initial_num_neg_label = num_negative_label

        # compute number of individuals per group
        num_pos_class_pos_lbl = int(num_positive_label * discrimination_factor)
        num_neg_class_pos_lbl = num_positive_label - num_pos_class_pos_lbl

        num_neg_class_neg_lbl = int(num_negative_label * discrimination_factor)
        num_pos_class_neg_lbl = num_negative_label - num_neg_class_neg_lbl

        # sample the corresponding number of individuals
        pos_cls_pos_lbl_samples = np.ones(num_pos_class_pos_lbl) * pos_class
        neg_cls_pos_lbl_samples = np.ones(num_neg_class_pos_lbl) * neg_class

        pos_cls_neg_lbl_samples = np.ones(num_pos_class_neg_lbl) * pos_class
        neg_cls_neg_lbl_samples = np.ones(num_neg_class_neg_lbl) * neg_class

        # sample the protected attribute
        # the top num_positive_label items belong all have a positive label
        # (the ltf_data is not shuffled)
        self._sens_attr = np.hstack((pos_cls_pos_lbl_samples,
                                     neg_cls_pos_lbl_samples,
                                     pos_cls_neg_lbl_samples,
                                     neg_cls_neg_lbl_samples))

        self._num_protected_individuals = num_neg_class_neg_lbl + num_neg_class_pos_lbl
        self._num_unprotected_individuals = num_pos_class_neg_lbl + num_pos_class_pos_lbl

    def _sample_initial_data(self, ):
        """Return sample of the initial data."""
        pos_samples = np.random.multivariate_normal(self.env_params.mean_pos,
                                                    self.env_params.cov_pos,
                                                    self._initial_num_pos_label)

        neg_samples = np.random.multivariate_normal(self.env_params.mean_neg,
                                                    self.env_params.cov_neg,
                                                    self._initial_num_neg_label)

        samples = np.vstack((pos_samples, neg_samples))
        labels = np.hstack((np.ones(self._initial_num_pos_label) * self.env_params.pos_label,
                            np.ones(self._initial_num_neg_label) * self.env_params.neg_label))

        return samples, self._sens_attr, labels

    def get_label(self, X_t, _X_sens_t=None):
        """Return the label for the points.

        Args:
             X_t (array_lika): 2D, the current features.
            _X_sens_t (array_like): 1D, the protected feature. Ignored here because
                the true label is independent if the protected feature.

        Returns:
            labels (numpy.ndarray): The labels.
        """
        distance_pos_mean = np.linalg.norm((X_t - self.env_params.mean_pos), axis=1)
        distance_neg_mean = np.linalg.norm((X_t - self.env_params.mean_neg), axis=1)

        labels = distance_pos_mean < distance_neg_mean

        int_labels = np.ones_like(labels) * self.env_params.neg_label
        int_labels[labels] = self.env_params.pos_label

        return int_labels

    def _direction_vector(self, x_i_t, offset):
        """Returns vector pointing from x_i_t to positive or negative cluster depending on offset.

        Args:
            x_i_t (array_like): 1D, the features of individual i at time t.
            offset (float): Sum of previous predictions.

        Returns:
            (array_like): Vector pointing from x_i_t to positive or negative mean.
        """
        if offset > 0:
            return self.env_params.pos_dir(x_i_t)
        else:
            return self.env_params.neg_dir(x_i_t)

    def _cov_function(self):
        """The covariance for the new point."""
        return self.env_params.local_variance

    def _sample_point(self, x_i_t, y_hat_i):
        """ Sample one new point.

        Args:
            x_i_t (array_like): 1D, the current features of individual i.
            y_hat_i (array_like): 1D, previous predictions for individual i.

        Returns:
            (numpy.ndarrary): The new point for individual i at time t+1.
        """
        cov = self._cov_function()
        mean = self._mean_function(x_i_t, y_hat_i)
        new_point = np.random.multivariate_normal(mean, cov, 1)
        return new_point.ravel()

    @abstractmethod
    def sample(self, X, _y, y_hat):
        """"""
        pass

    @abstractmethod
    def _mean_function(self, x_i_t, y_hat_i):
        """"""
        pass
