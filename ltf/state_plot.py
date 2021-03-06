"""2D Scatter plot for data points of one iteration."""
import numpy as np


class StatePlot:
    """Plot data of one time step."""

    def __init__(self, x_lim=None, y_lim=None, pos_lbl=1, neg_lbl=0, pos_cls=1, neg_cls=0):
        """

        Args:
            x_lim [int, int]: Min and max value for x-axis of plot.
            y_lim [int, int]: Min and max value for y-axis of plot.
            pos_cls (int): The positive class of the protected attribute.
            neg_cls (int): The negative class of the protected attribute.
            pos_lbl (int): The positive label.
            neg_lbl (int): The negative label.
        """

        self._pos_label = pos_lbl
        self._neg_label = neg_lbl

        self._pos_class = pos_cls
        self._neg_class = neg_cls

        self._x_lim = x_lim
        self._y_lim = y_lim

    def get_decision_boundary(self, X_t, predict_func, num_points=500):
        """Return triple representing the decision boundary generated by predict_func.

        Args:
            X_t (optional): Dataset, used to estimate min and max values if _x_lim and _y_lim are None.
            predict_func (function): The classifiers prediction function, {X, X_sense} -> R.
            num_points (int): Number of points used for mesh grid.

        Returns:
            xx, yy, Z (numpy.ndarray, numpy.ndarray, numpy.ndarray): The decision boundary.
                2D arrays of shape [num_points x num_points]. xx and yy is the meshgrid and
                Z predictions for the meshgrid reshaped to shape of xx.
        """

        if self._x_lim is not None and self._y_lim is not None:
            x1_min, x1_max = self._x_lim
            x2_min, x2_max = self._y_lim
        else:
            x1_min, x1_max = X_t[:, 0].min() - 3, X_t[:, 0].max() + 3
            x2_min, x2_max = X_t[:, 1].min() - 3, X_t[:, 1].max() + 3

        x1_step = (x1_max - x1_min) / num_points
        x2_step = (x2_max - x2_min) / num_points

        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, x1_step),
                             np.arange(x2_min, x2_max, x2_step))

        mesh = np.c_[xx.ravel(), yy.ravel()]

        # TODO: generate protected attribute in plausible way
        X_sens_dummy = np.zeros(np.shape(mesh)[0])

        Z = predict_func(mesh, X_sens_dummy)
        Z = Z.reshape(xx.shape)

        return xx, yy, Z

    @staticmethod
    def add_decision_boundary(ax, D, cmap=None, label=""):
        """Add the decision boundary to the plot axis ax.

        Args:
            ax (matplotlib.pyplot.axes): Axis to add boundary to.
            D (numpy.ndarray, numpy.ndarray, numpy.ndarray): The decision boundary as returned above.
            cmap (str): Colormap, https://matplotlib.org/tutorials/colors/colormaps.html.
            label (str): Label for the boundary.
        """

        xx, yy, Z = D

        if cmap is not None:
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=.5)
        else:
            CS = ax.contour(xx, yy, Z)
            CS.collections[0].set_label(label)

    def scatter_data_points(self, ax, X_t, y_hat_t, X_sensitive_t, title="", print_stats=False):
        """Scatter plot points (X_t, y_hat_t, X_sensitive_t) at axis ax.

        Args:
            ax (matplotlib.pyplot.axes): Axis to scatter points.
            X_t (array_like): 2D, features at time step t.
            y_hat_t (array_like): 1D, predictions at time step t.
            X_sensitive_t (array_like): 1D. sensitive features at time step t.
            title (str): Label for the axis.
            print_stats (bool): If true, stats are printed in each call.
        """

        pos_lbl_mask = y_hat_t == self._pos_label
        neg_lbl_mask = y_hat_t == self._neg_label

        pos_cls_mask = X_sensitive_t == self._pos_class
        neg_cls_mask = X_sensitive_t == self._neg_class

        pos_lbl_pos_cls = np.logical_and(pos_lbl_mask, pos_cls_mask)
        ax.scatter(X_t[pos_lbl_pos_cls, 0],
                   X_t[pos_lbl_pos_cls, 1],
                   label="pos label and class " + str(np.sum(pos_lbl_pos_cls)),
                   marker="x",
                   c="green")

        pos_lbl_neg_cls = np.logical_and(pos_lbl_mask, neg_cls_mask)
        ax.scatter(X_t[pos_lbl_neg_cls, 0],
                   X_t[pos_lbl_neg_cls, 1],
                   label="pos label and neg class " + str(np.sum(pos_lbl_neg_cls)),
                   marker="o",
                   c="darkgreen")

        neg_lbl_pos_cls = np.logical_and(neg_lbl_mask, pos_cls_mask)
        ax.scatter(X_t[neg_lbl_pos_cls, 0],
                   X_t[neg_lbl_pos_cls, 1],
                   label="neg label and pos class " + str(np.sum(neg_lbl_pos_cls)),
                   marker="x",
                   c="darkred")

        neg_lbl_neg_cls = np.logical_and(neg_lbl_mask, neg_cls_mask)
        ax.scatter(X_t[neg_lbl_neg_cls, 0],
                   X_t[neg_lbl_neg_cls, 1],
                   label="neg label and class " + str(np.sum(neg_lbl_neg_cls)),
                   marker="o",
                   c="red")

        if print_stats:
            txt = "number of positive labels:" + str(np.sum(pos_lbl_mask)) + \
                  "\nnumber of negative labels: " + str(np.sum(neg_lbl_mask)) + "\n" + \
                  "\npositive label positive class: " + str(np.sum(pos_lbl_pos_cls)) + \
                  "\npositive label negative class: " + str(np.sum(pos_lbl_neg_cls)) + \
                  "\nnegative label positive class: " + str(np.sum(neg_lbl_pos_cls)) + \
                  "\nnegative label negative class: " + str(np.sum(neg_lbl_neg_cls))

            print(txt)

        txt = "num positive labels :" + str(np.sum(pos_lbl_mask)) + \
              "\nnum negative labels: " + str(np.sum(neg_lbl_mask)) + "\n"

        if self._x_lim is not None and self._y_lim is not None:
            ax.set_xlim(self._x_lim)
            ax.set_ylim(self._y_lim)

        ax.set_title(title + "\n" + txt)
