# Student project: a dynamic environment for static decision functions
This project provides a framework to visualize the impact of static fairness-aware machine learning algorithms in 
dynamic environments. It is built around the [AIF360](https://github.com/IBM/AIF360) toolkit to provide out-of-the-box 
access to numerous fairness metrics and algorithms. The main idea is to repeatedly apply the same decision 
function to dynamic data and visualize the resulting outcomes in form of plots. 

The backbone of the framework is a loop that repeatedly samples data ```X_t, y_t``` for a certain number of steps:

1. Sample the data of the current time step as ```X_t, y_t ~ P(X_t-1, y_hat_t-1)``` with shapes ``(n, 2 )`` and ``(n, )``
1. Run a fair classification algorithm from AIF360 to compute binary predictions ```y_hat_t``` with shape ``(n, )``
1. Compute some real valued fairness metrics from AIF360 on the data ```X_t, y_t, y_hat_t```

After the last iteration, the computed metrics are plotted through time.

A short example is given below. Detailed [examples](https://github.com/danilobr94/dynamic-aif-framework/tree/master/examples) 
are available as jupyter-notebooks. The source code is located in the folder [code](https://github.com/danilobr94/dynamic-aif-framework/tree/master/code).
The final report is available [here](https://github.com/danilobr94/dynamic-aif-framework/blob/master/examples/report.pdf). 

## Dynamics
Two different dynamics are implemented as two data generators. Overall it is assumed that a positive decision will 
have a positive effect (e.g. being accepted for a job will have a positive impact on the qualification for future jobs). 
In the ``individual`` data generator only the person receiving the positive decision 
will benefit (e.g. only the one being accepted will benefit). In the ``group`` data generator it is assumed that a 
positive decision will also have a positive impact on the social environment of that person 
(e.g. the person being accepted serves as role model for the community it belongs to).
Details about the dynamics and implementation can be found in the report.

## Notes
This is a simple environment that was used to investigate some baseline for another project currently in progress.
It is similar to [ml-fairness-gym](https://github.com/google/ml-fairness-gym) with the main difference being
that ml-fairness-gym is built for re-enforcement agents and this framework for classification.

# Short Summary
- Two sequential data generators provide sequential data.
	1. Assuming a positive influence of a decision on the whole group with the same attribute.
	1. Assuming only the individual itself benefits from a positive decision.

- Plot generator runs a loop: sampling data, running a classifier, evaluating metrics in each iteration and finally plotting the results.
- Interfaces to use [AIF360](https://github.com/IBM/AIF360) metrics and classifiers.  

# Short Example

Metrics from AIF360:
```python
metric = AifLongTermMetric(["accuracy", "base_rate"])
```

Fair algorithm from AIF360: 
```python
clf = AifLongTermCLF(aif360.algorithms.inprocessing.PrejudiceRemover())
```

Data Generator:
```python
generator = IndividualDataGenerator()
```

Running the plot generator:
```python
ltf_plt = LongTermFairnessPlot(generator, clf, metric.metric)
ltf_plt.run(20)
```

Results:
```python
ltf_plt.plot_ltf(metric._metrics)
```

![Results](https://github.com/danilobr94/dynamic-aif-framework/blob/master/ltf/example_plot.png)
