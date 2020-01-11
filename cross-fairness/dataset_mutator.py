"""Random changes to ltf_data set"""

from aif360.datasets.structured_dataset import StructuredDataset
import numpy as np


class DataMutator:

    def __init__(self, dataset: StructuredDataset):
        """"""
        self.dataset = dataset

    def shuffle_attribute(self, ind):
        """Randomly shuffle the values of an attribute"""
        dataset_new = self.dataset.copy(deepcopy=True)

        new_values = self.dataset.features[ind]
        np.random.shuffle(new_values)

        dataset_new.features[ind] = new_values

        return dataset_new

    def randomize_attribute(self, ind, random_generator=None):
        """Randomly assign new values to the attribute"""

        dataset_new = self.dataset.copy(deepcopy=True)
        feature_size = dataset_new.features.shape[0]

        if random_generator is None:
            random_generator = lambda n: np.random.normal(np.mean(dataset_new.features[ind]), n)
        dataset_new.features[ind] = random_generator(feature_size)

        return dataset_new

    def substitute_group(self, ind, group, new_group):
        """Remove group from a categorical feature"""
        dataset_new = self.dataset.copy(deepcopy=True)

        new_values = dataset_new.features[ind]
        group_indices = new_values == group
        new_values[group_indices] = new_group

        dataset_new.features[ind] = new_values

        return dataset_new

    # ## just for fun
    def one_point_crossover(self, dataset_two: StructuredDataset, ind):
        """"""
        assert self.dataset.features.shape == dataset_two.features.shape

        dataset_new = self.dataset.copy(deepcopy=True)

        feature_size = dataset_two.features.shape[0]
        cross_over_point = np.random.randint(0, high=feature_size)

        new_values = np.zeros(feature_size)
        new_values[:cross_over_point] = self.dataset.features[:cross_over_point]
        new_values[cross_over_point:] = self.dataset_two.features[cross_over_point:]

        dataset_new.features[ind] = new_values

        return dataset_new

    def k_point_crossover(self, dataset_two: StructuredDataset, k=2):
        """"""
        raise NotImplemented
