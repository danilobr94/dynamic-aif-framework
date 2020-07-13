"""Base class for functions using aif360. Provides function to convert data to aif360 data frame."""
from abc import ABC
import numpy as np
import pandas as pd
from aif360.datasets.binary_label_dataset import BinaryLabelDataset


class AifLongTermBase(ABC):
    """"""

    def __init__(self, pos_class=1, neg_class=0, pos_label=1, neg_label=0):
        """

        Args:
            pos_class (int): The positive class of the protected attribute.
            neg_class (int): The negative class of the protected attribute.
            pos_label (int): The positive label.
            neg_label (int): The negative label.
        """

        self._pos_label = pos_label
        self._neg_label = neg_label
        self._neg_class = neg_class
        self._pos_class = pos_class

        self.label_name = "label"
        self.protected_attribute_name = "protected"
        self.feature_col_names = "feature_"

    def _to_aif_data_frame(self, X_t, X_sens_t, y_t=None):
        """Stacks attributes into aif360 dataset.

        Args:
            X_t (array_like): 2D, the unprotected features.
            X_sens_t (array_like): 1D, the protected attribute.
            y_t (array_like): The labels. If None, all labels are set to 0.

        Returns:
            dataset (aif360.BinaryLabelDataset): Inputs stacked into aif360 data frame.
        """

        columns = [self.feature_col_names + str(s) for s in range(np.shape(X_t)[1])]
        columns.append(self.protected_attribute_name)
        columns.append(self.label_name)

        if y_t is None:
            y_t = np.zeros(np.shape(X_t)[0])

        d = np.column_stack((X_t, X_sens_t, y_t))

        df = pd.DataFrame(d, columns=columns)

        dataset = BinaryLabelDataset(favorable_label=self._pos_label,
                                     unfavorable_label=self._neg_label,
                                     df=df,
                                     label_names=[self.label_name],
                                     protected_attribute_names=[self.protected_attribute_name],
                                     unprivileged_protected_attributes=[[self._neg_class]],
                                     privileged_protected_attributes=[[self._pos_class]])

        return dataset
