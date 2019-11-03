from aif360.datasets.german_dataset import GermanDataset
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aif360.metrics import ClassificationMetric
from methods.cross_fairness import CrossFairness


def define_metric(privileged_groups, unprivileged_groups):
    """"""

    def _metric(dataset, predicted_dataset):
        m = ClassificationMetric(dataset, predicted_dataset, privileged_groups, unprivileged_groups)
        return {"accuracy ": m.accuracy(), "disparate impact ": m.disparate_impact()}

    return _metric


tau = 0.8
model = MetaFairClassifier(tau=tau, sensitive_attr="sex")

gcd = GermanDataset()
print(gcd.feature_names)
print(gcd.features[0])

met = define_metric([{'sex': 1}], [{'sex': 0}])

cf = CrossFairness(gcd, model, met)

cf.randomize_feature(0)
cf.shuffle_feature(0)
cf.remove_feature(0)
cf.remove_group(0, 4)
cf.substitute_group(0, 4, 5)

cf.plot_evaluation()
