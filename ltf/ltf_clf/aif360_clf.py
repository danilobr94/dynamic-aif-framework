"""Interface for aif360 algorithms in long term simulation."""
from ltf.ltf_aif_base import AifLongTermBase


class AifLongTermCLF(AifLongTermBase):
    """Wrapper class to  use aif360 algorithms.

    Stacks inputs into aif360 data frame and calls the classifier passed on initialization.
    """

    def __init__(self, clf, pos_class=1, neg_class=0, pos_label=1, neg_label=0):
        """

        Args:
            clf (aif360.algorithms.Transformer): Classifier.
            pos_class (int): The positive class of the protected attribute.
            neg_class (int): The negative class of the protected attribute.
            pos_label (int): The positive label.
            neg_label (int): The negative label.
        """
        self._clf = clf
        super().__init__(pos_class=pos_class, neg_class=neg_class,
                         pos_label=pos_label, neg_label=neg_label)

    def fit(self, X_t, X_sens_t, y_t):
        """Fit the clf on data at time step t."""
        dataset = self._to_aif_data_frame(X_t, X_sens_t, y_t)
        self._clf.fit(dataset)

    def predict(self, X_t, X_sens_t):
        """Run predictions on time step t."""
        dataset = self._to_aif_data_frame(X_t, X_sens_t, None)
        predicted_dataset = self._clf.predict(dataset)
        return predicted_dataset.labels.ravel()
