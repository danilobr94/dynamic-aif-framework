from aif360.metrics import ClassificationMetric
from .aif_base import AifLongTermBase


class AifLongTermMetric(AifLongTermBase):
    """"""

    def metric(self, X, X_sens, y, y_hat):
        """"""
        dataset = self._to_aif_data_frame(X, X_sens, y)
        classified_dataset = self._to_aif_data_frame(X, X_sens, y_hat)

        unprivileged = {self.protected_attribute_name: self._neg_class}
        privileged = {self.protected_attribute_name: self._pos_class}

        metric = ClassificationMetric(dataset, classified_dataset,
                                      unprivileged_groups=[unprivileged],
                                      privileged_groups=[privileged])

        ret = []

        #ret.append(metric.equal_opportunity_difference())
        ret.append(metric.accuracy())
        ret.append(metric.disparate_impact())

        return ret


class AifLongTermPrediction(AifLongTermBase):
    """"""

    def __init__(self, clf, pos_class=1, neg_class=0):
        """"""
        self._clf = clf
        super().__init__(pos_class=pos_class, neg_class=neg_class)

    def fit(self, X, X_sens, y):
        """"""
        dataset = self._to_aif_data_frame(X, X_sens, y)
        self._clf.fit(dataset)

    def predict(self, X, X_sens, y):
        """"""
        dataset = self._to_aif_data_frame(X, X_sens, y)
        predicted_dataset = self._clf.predict(dataset)
        return predicted_dataset.labels.ravel()
