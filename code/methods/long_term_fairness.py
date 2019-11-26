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
        self.init_data()

        for _i in range(num_steps):
            self.run_generation()

    def init_data(self):
        """"""
        X_init, X_sens_init, y_init = self._sampling_function(None, None, None)
        self._clf.fit(X_init, X_sens_init, y_init)
        y_hat_init = self._clf.predict(X_init, X_sens_init)

        self._X.append(X_init)
        self._y.append(y_init)
        self._y_hat.append(y_hat_init)

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

    def plot_generation(self):
        """"""
        m1 = self._y[-1] == 1
        m2 = self._y[-1] == 0

        plt.scatter(self._X[-1][m1, 0], self._X[-1][m1, 1], label="positive")
        plt.scatter(self._X[-1][m2, 0], self._X[-1][m2, 1], label="negative")

        plt.title("Generation " + str(len(self._y_hat)-1))

        txt = "number in positive class:" + str(np.sum(m1)) + \
              "\nnumber in negative class: " + str(np.sum(m2))
        plt.text(6, -2, txt)

        plt.legend()
        plt.show()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from individual_data_generator_simple import DataGenerator

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

    class df():
        def __init__(self):
            self.clf = LogisticRegression()

        def fit(self, X, s, y):
            self.clf.fit(X, y)

        def predict(self, X, _s):
            return self.clf.predict(X)

        def score(self,  X, s, y):
            self.clf.score(X, y)

    generator = DataGenerator()

    l = LongTermFairnessPlot(generator.generate_data, df(), metric, update_clf=False)

    l.init_data()
    l.plot_generation()
    for _ in range(10):

        l.run_generation()
        l.plot_generation()

    l.plot(["accuracy", "disparate impact"])
