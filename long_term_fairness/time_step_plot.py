import numpy as np


class GenerationPlot:

    def __init__(self, x_lim, y_lim, pos_lbl, neg_lbl, pos_cls, neg_cls):
        """"""

        self._pos_label = pos_lbl
        self._neg_label = neg_lbl

        self._pos_class = pos_cls
        self._neg_class = neg_cls

        self._x_lim = x_lim
        self._y_lim = y_lim

    def get_decision_boundary(self, X_t, predict_func, num_points=500):
        """Get tripe representing the decision boundary generated by predict_func.

        Args:
            X_t (optional): dataset, used to estimate min max values if _x_lim and _y_lim are None.
            predict_func:
            num_points:

        Returns:
            xx, yy, Z
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
            ax: matplotlib axes
            D: triple (xx, yy, Z) the decision boundary
            cmap (str): colormap
            label (str): boundary label

        """

        xx, yy, Z = D

        if cmap is not None:
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=.5)
        else:
            CS = ax.contour(xx, yy, Z)
            CS.collections[0].set_label(label)

    def scatter_data_points(self, ax, X_t, y_hat_t, X_sensitive_t, title="", print_stats=False):
        """

        Args:
            ax:
            X_t:
            y_hat_t:
            title:
            print_stats:

        Returns:

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