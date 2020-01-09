import numpy as np
import pandas as pd
from abc import ABC
from aif360.datasets.binary_label_dataset import BinaryLabelDataset


class AifLongTermBase(ABC):

    def __init__(self, pos_class=1, neg_class=0, pos_label=1, neg_label=0):
        """"""

        self.pos_label = pos_label
        self._neg_label = neg_label
        self._neg_class = neg_class
        self._pos_class = pos_class

        self.label_name = "label"
        self.protected_attribute_name = "protected"
        self.feature_col_names = "feature_"

    def _to_aif_data_frame(self, X, x_sense, y=None):
        """

        Args:
            X: 2D, the unprotected features
            x_sense: the protected attribute
            y: the labels, if None all labels are set to 0

        Returns:
            AIF360.BinaryLabelDataset
        """

        feature_cols = [self.feature_col_names + str(s) for s in range(np.shape(X)[1])]

        if y is None:
            y = np.zeros(np.shape(X)[0])

        columns = np.concatenate((feature_cols, (self.protected_attribute_name, self.label_name)))
        d = np.column_stack((X, x_sense, y))

        df = pd.DataFrame(d, columns=columns)

        dataset = BinaryLabelDataset(favorable_label=self.pos_label,
                                     unfavorable_label=self._neg_label,
                                     df=df,
                                     label_names=[self.label_name],
                                     protected_attribute_names=[self.protected_attribute_name],
                                     unprivileged_protected_attributes=[[self._neg_class]],
                                     privileged_protected_attributes=[[self._pos_class]])

        return dataset
