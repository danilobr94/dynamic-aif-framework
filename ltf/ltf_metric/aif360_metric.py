"""Interface for aif360 metric in long term simulation."""
from aif360.metrics import ClassificationMetric
from ltf.ltf_aif_base import AifLongTermBase


class AifLongTermMetric(AifLongTermBase):
    """Wrapper class to use aif360 metrics for LongTerm framework."""

    def __init__(self, metrics=("accuracy", "base_rate"),
                 pos_class=1, neg_class=0, pos_label=1, neg_label=0):
        """

        Args:
            metrics (list(str)): Name of aif360.ClassificationMetric functions to be applied.
            pos_class (int): The positive class of the protected attribute.
            neg_class (int): The negative class of the protected attribute.
            pos_label (int): The positive label.
            neg_label (int): The negative label.
        """
        self._metrics = metrics

        super().__init__(pos_class=pos_class, neg_class=neg_class,
                         pos_label=pos_label, neg_label=neg_label)

    def metric(self, X_t, X_sens_t, y_t, y_hat_t):
        """Compute metrics for the current iteration.

        Transforms input parameters into aif360 data frame and computes metrics.

        Args:
            X_t (array_like): 2D, features at time step t.
            X_sens_t (array_like): 1D, the protected attribute at time step t.
            y_t (array_like): 1D, true labels at time step t.
            y_hat_t (array_like): 1D, predictions at time step t.

        Returns:
            list(float): List of floats representing metrics from self._metrics.
        """
        dataset = self._to_aif_data_frame(X_t, X_sens_t, y_t)
        classified_dataset = self._to_aif_data_frame(X_t, X_sens_t, y_hat_t)

        unprivileged = {self.protected_attribute_name: self._neg_class}
        privileged = {self.protected_attribute_name: self._pos_class}

        clf_metric = ClassificationMetric(dataset, classified_dataset,
                                          unprivileged_groups=[unprivileged],
                                          privileged_groups=[privileged])

        ret = []

        for metric_name in self._metrics:

            try:
                aif_metric = getattr(clf_metric, metric_name)
                ret.append(aif_metric())

            except AttributeError:
                print("Metric " + metric_name + " not found for object " + str(type(clf_metric)))

        return ret
