"""Visualize long term effect of a classifier

Steps:
    1. Make predictions using decision function on data set
    2. Evaluate accuracy and fairness metric on predictions
    3. Sample next generation using sampling function
    4. Repeat step 1

    TODO: pass initial data set?
    TOD0: aif360

"""
import matplotlib.pyplot as plt


class LongTermFairnessPlot:
    """"""

    def __init__(self, sampling_function, clf, fairness_metric, update_clf=False):
        """

        :param sampling_function: (function) X, y, y_hat
        :param clf: (object) must implement fit(X, s, y) and predict(X, s) functions
        :param fairness_metric: (function) X, s, y, y_hat
        :param sensitive_attributes: (ndarray)
        :param update_clf: (bool)
        """
        self._sampling_function = sampling_function
        self._clf = clf
        self._fairness_metric = fairness_metric

        self._update_clf = update_clf

        self._results = []
        self._X = []
        self._y = []
        self._y_hat = []

    def export_data(self):
        """"""
        raise NotImplemented

    def _fit_clf(self):
        """"""
        X = np.vstack(self._X).squeeze()
        y = np.hstack(self._y).squeeze()

        num_repetitions = np.shape(self._X)[0]
        s = np.repeat(self._sensitive_attributes, num_repetitions)

        self._clf.fit(X, s, y)

    def run(self, num_steps):
        """"""
        X_init, X_sens_init, y_init = self._sampling_function(None, None, None)
        self._clf.fit(X_init, X_sens_init, y_init)

        for _i in range(num_steps):
            self.run_generation()

    def run_generation(self):
        """"""
        X_t, X_sens_t, y_t = self._sampling_function(self._X, self._y, self._y_hat)
        y_hat_t = self._clf.predict(X_t, X_sens_t)

        metric = self._fairness_metric(X_t, X_sens_t, y_t, y_hat_t)

        self._X.append(X_t)
        self._y.append(y_t)
        self._y_hat.append(y_hat_t)

        if self._update_clf:
            self._fit_clf()

        self._results.append(metric)

        return metric

    def plot(self, labels=""):
        """"""
        result_arr = np.asarray(self._results)
        num_generations, num_metrics = result_arr.shape

        for i in range(num_metrics):
            lbl = "metric " + str(i) if labels == "" else labels[i]
            plt.plot(range(num_generations), result_arr[:, i], label=lbl)

        plt.xlabel("Generation")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def metric(x, s, y, y_hat):
        ret = []

        ret.append(1 -(np.sum(abs(y-y_hat))/len(y)))

        pos_cls = s == 1
        neg_cls = s == 0
        pos_lbl = y_hat == 1

        pos_cls_pos_lbl = pos_cls == pos_lbl
        neg_cls_pos_lbl = neg_cls == pos_lbl

        p = np.sum(pos_cls_pos_lbl)
        n = np.sum(neg_cls_pos_lbl)

        dsp_im = p/n if (p/n) < 1 else n/p

        ret.append(dsp_im)

        return ret

    class df:
        def __init__(self):
            self.clf = LogisticRegression()

        def fit(self, X, s, y):
            self.clf.fit(X, y)

        def predict(self, X, _s):
            return self.clf.predict(X)


    sens_attrs = np.random.randint(0, 2, size=1000)

    def sf(X, y, y_hat, num_pos=500, num_neg=500):
        """Sample points from multivariate gaussian"""
        mean_1 = [8, 5]
        cov_1 = [[1.2, 0], [0, 1.3]]

        mean_2 = [0, 1]
        cov_2 = [[1, 0], [0, 1.2]]
        pos_samples = np.random.multivariate_normal(mean_1, cov_1, num_pos)
        neg_samples = np.random.multivariate_normal(mean_2, cov_2, num_neg)

        samples = np.vstack((pos_samples, neg_samples))
        labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))

        return samples, sens_attrs, labels

    def sf_hist(X, y, y_hat, num_pos=500, num_neg=500):
        mean_1 = [8, 5]
        cov_1 = [[1.2, 0], [0, 1.3]]

        mean_2 = [0, 1]
        cov_2 = [[1, 0], [0, 1.2]]
        pos_samples = np.random.multivariate_normal(mean_1, cov_1, num_pos)
        neg_samples = np.random.multivariate_normal(mean_2, cov_2, num_neg)

        samples = np.vstack((pos_samples, neg_samples))
        labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))

        return samples, sens_attrs, labels

    import matplotlib.pyplot as plt

    s, _, l = sf(None, None, None)
    m1 = l == 1
    m2 = l == 0

    plt.scatter(s[m1, 0], s[m1, 1])
    plt.scatter(s[m2, 0], s[m2, 1])
    plt.show()

    l = LongTermFairnessPlot(sf, df(), metric, update_clf=False)
    l.run(100)
    l.plot(["accuracy", "disparate impact"])
