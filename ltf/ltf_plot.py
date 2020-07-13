"""Run  loop simulating long term effect of a classifier and plot final results.

Steps:
    1. Make predictions using decision function on data set
    2. Evaluate metrics on predictions
    3. Sample next generation using sampling function
    4. Repeat step 1

TODO:
    * pass initial ltf_data set?
    * add a separate baseline_clf?
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from .state_plot import StatePlot
from .ltf_data.data_generator_base import DataGeneratorBase
from .ltf_clf.aif360_clf import AifLongTermCLF


class _States:
    """Class to hold data of all time steps."""

    def __init__(self):
        """"""
        self.X = []  # type: List[np.ndarray]
        self.X_sensitive = []  # type: List[np.ndarray]
        self.y = []  # type: List[np.ndarray]
        self.y_hat = []  # type: List[np.ndarray]
        self.metric = []  # type: List[np.ndarray]

    def append(self, X_t, X_s_t, y_t, y_hat_t, metric=None):
        """Append data of one time step to states."""
        self.X.append(X_t)
        self.X_sensitive.append(X_s_t)
        self.y.append(y_t)
        self.y_hat.append(y_hat_t)
        if metric is not None:
            self.metric.append(metric)


class LongTermFairnessPlot:
    """"""

    def __init__(self, data_generator, clf, fairness_metric, update_clf=False, x_lim=None, y_lim=None):
        """

        Args:
            data_generator (DataGeneratorBase): Must implement sample(X, y, y_hat) and get_label(X) functions.
            clf (AifLongTermCLF): Must implement fit(X, X_s, y) and predict(X, X_s) functions.
            fairness_metric (function): {X, X_s, y, y_hat} -> R.
            update_clf (bool): If true, clf is updated every iteration.
        """
        self._data_generator = data_generator

        self._clf = clf
        self._fairness_metric = fairness_metric

        self._update_clf = update_clf

        self.true_states = _States()
        self.baseline_states = _States()

        self._pos_label = 1
        self._neg_label = 0

        self._pos_class = 1
        self._neg_class = 0

        self._x_lim = x_lim
        self._y_lim = y_lim

        self._data_generating_decision_boundary = None
        self._clf_decision_boundary = None

        self._state_plotter = StatePlot(self._x_lim,
                                        self._y_lim,
                                        self._pos_label, self._neg_label,
                                        self._pos_class, self._neg_class)

    def run(self, num_steps):
        """Run num steps iterations of true and baseline generation."""
        self.init_data()

        for _i in range(num_steps):
            self.run_step()
            self.run_baseline_step()

    def init_data(self):
        """Sample the initial data."""
        X_init, X_sens_init, y_init = self._data_generator.sample(None, None, None)

        self._fit_clf(X_init, X_sens_init, y_init)
        y_hat_init = self._clf.predict(X_init, X_sens_init)

        self.true_states.append(X_init, X_sens_init, y_init, y_hat_init)
        self.baseline_states.append(X_init, X_sens_init, y_init, y_hat_init)

    def run_step(self):
        """Run one generation of the true data pipeline."""
        X_t, X_sens_t, y_t = self._data_generator.sample(self.true_states.X,
                                                         self.true_states.y,
                                                         self.true_states.y_hat)

        y_hat_t = self._clf.predict(X_t, X_sens_t)
        metric = self._fairness_metric(X_t, X_sens_t, y_t, y_hat_t)

        self.true_states.append(X_t, X_sens_t, y_t, y_hat_t, metric)

        if self._update_clf:
            self._fit_clf()

        return metric

    def run_baseline_step(self):
        """Run one iteration of baseline pipeline and append metric to results.

        In the baseline generation all previous predictions are assumed to be positive.
        """
        y_hat_pos = np.ones(np.shape(self.baseline_states.y_hat)) * self._pos_label

        X_t_base, X_sens_t_base, y_t_base = self._data_generator.sample(self.baseline_states.X,
                                                                        self.baseline_states.y,
                                                                        y_hat_pos)

        y_hat_t_base = self._clf.predict(X_t_base, X_sens_t_base)
        metric = self._fairness_metric(X_t_base, X_sens_t_base, y_t_base, y_hat_t_base)

        self.baseline_states.append(X_t_base, X_sens_t_base, y_t_base, y_hat_t_base, metric)

        return metric

    def plot_ltf(self, labels=""):
        """Plot the results for both pipelines over all generations."""
        result_arr = np.asarray(self.true_states.metric)
        baseline_result_arr = np.asarray(self.baseline_states.metric)
        num_generations, num_metrics = result_arr.shape

        for i in range(num_metrics):
            lbl = "ltf_clf " + str(i) if labels == "" else labels[i]
            plt.plot(range(num_generations), result_arr[:, i], label=lbl)
            plt.plot(range(num_generations),
                     baseline_result_arr[:, i],
                     label="baseline " + lbl,
                     linestyle="--")

        plt.xlabel("Generation")
        plt.legend()
        plt.show()

    def plot_step(self, plot_decision_boundary=True):
        """Scatter plot the points of one iteration. """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # maybe add decision boundary
        if plot_decision_boundary:
            if self._data_generating_decision_boundary is None:
                self._data_generating_decision_boundary = \
                    self._state_plotter.get_decision_boundary(self.true_states.X[-1],
                                                              self._data_generator.get_label)

            if self._clf_decision_boundary is None or self._update_clf:
                self._clf_decision_boundary = self._state_plotter.get_decision_boundary(self.true_states.X[-1],
                                                                                        self._clf.predict)

            self._state_plotter.add_decision_boundary(ax1, self._clf_decision_boundary, label="clf boundary")
            self._state_plotter.add_decision_boundary(ax2, self._clf_decision_boundary, label="clf boundary")

            self._state_plotter.add_decision_boundary(ax1, self._data_generating_decision_boundary, cmap="Pastel1")
            self._state_plotter.add_decision_boundary(ax2, self._data_generating_decision_boundary, cmap="Pastel1")

        # scatter the points
        self._state_plotter.scatter_data_points(ax1,
                                                self.true_states.X[-1],
                                                self.true_states.y_hat[-1],
                                                self.true_states.X_sensitive[-1],
                                                "true ltf_data")

        self._state_plotter.scatter_data_points(ax2,
                                                self.baseline_states.X[-1],
                                                self.baseline_states.y_hat[-1],
                                                self.baseline_states.X_sensitive[-1],
                                                "baseline ltf_data")

        fig.suptitle("Generation " + str(len(self.true_states.y_hat)-1))

        ax1.legend()
        ax2.legend()

        plt.show()

    def _fit_clf(self, X=None, X_s=None, y=None):
        """Fit clf to passed data or to the whole data set if None."""

        if X is None or X_s is None or y is None:
            X = np.vstack(self.true_states.X).squeeze()
            y = np.hstack(self.true_states.y).squeeze()
            X_s = np.hstack(self.true_states.X_sensitive)

        self._clf.fit(X, X_s, y)
