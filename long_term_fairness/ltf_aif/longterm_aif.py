from aif360.metrics import ClassificationMetric
from .longterm_aif_base import AifLongTermBase


class AifLongTermMetric(AifLongTermBase):
    """Use AIF360 metrics to for LongTerm framework."""

    def __init__(self, metrics=["accuracy", "disparate_impact"],
                 pos_class=1, neg_class=0, pos_label=1, neg_label=0):
        """

        Args:
            metrics (list(str)): name of AIF360.ClassificationMetric class function
            pos_class:
            neg_class:
            pos_label:
            neg_label:
        """
        self._metrics = metrics
        super().__init__(pos_class=pos_class, neg_class=neg_class,
                         pos_label=pos_label, neg_label=neg_label)

    def metric(self, X, X_sens, y, y_hat):
        """"""
        dataset = self._to_aif_data_frame(X, X_sens, y)
        classified_dataset = self._to_aif_data_frame(X, X_sens, y_hat)

        unprivileged = {self.protected_attribute_name: self._neg_class}
        privileged = {self.protected_attribute_name: self._pos_class}

        clf_metric = ClassificationMetric(dataset, classified_dataset,
                                          unprivileged_groups=[unprivileged],
                                          privileged_groups=[privileged])

        ret = []

        for metric_name in self._metrics:
            try:
                metric = getattr(clf_metric, metric_name)
                ret.append(metric())
            except AttributeError:
                print("Metric " + metric_name + " not found for object " + str(type(clf_metric)))

        return ret


class AifLongTermPrediction(AifLongTermBase):
    """Use AIF360 predictors for LongTerm framework."""

    def __init__(self, clf, pos_class=1, neg_class=0, pos_label=1, neg_label=0):
        """"""
        self._clf = clf
        super().__init__(pos_class=pos_class, neg_class=neg_class,
                         pos_label=pos_label, neg_label=neg_label)

    def fit(self, X, X_sens, y):
        """"""
        dataset = self._to_aif_data_frame(X, X_sens, y)
        self._clf.fit(dataset)

    def predict(self, X, X_sens):
        """"""
        dataset = self._to_aif_data_frame(X, X_sens, None)
        predicted_dataset = self._clf.predict(dataset)
        return predicted_dataset.labels.ravel()
