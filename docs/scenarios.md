# Verified scenario results

These numbers come from `python -m ml_model_dojo.scenarios.verify`, which runs every scenario with the settings stored in `scenarios/definitions.py`. Timings depend on the machine. Each follow-up row applies the extra setting that the scenario text asks the student to change.

## 1. Linear is enough

Data: `blobs_sep`, n = 300, noise = 0.3, minority share = 0.5, test mode = random, seed = 1.

What to look for: Every model reaches nearly the same test accuracy, so the simplest and fastest model (logistic regression) is the right choice.

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 1.000 | 1.000 | 1.000 | 240 |
| kNN | 1.000 | 1.000 | 1.000 | 5 |
| Decision tree | 1.000 | 1.000 | 1.000 | 2 |
| Random forest | 1.000 | 1.000 | 1.000 | 378 |
| Gradient boosting | 1.000 | 1.000 | 1.000 | 1834 |
| SVM | 1.000 | 1.000 | 1.000 | 5 |
| Neural network | 1.000 | 1.000 | 1.000 | 2305 |

Majority class baseline on the test set: 0.500.

Forest test loss: 0.000 at best, final 0.000 at tree 100.

Boosting test loss: minimum 0.009 at tree 25, final value 0.009 at tree 100 (+0 percent).

## 2. XOR breaks linear models

Data: `xor`, n = 600, noise = 0.3, minority share = 0.5, test mode = random, seed = 1.

What to look for: Logistic regression sits near 50 percent while the others do well. Then open the Linear tab (Edit), set the polynomial degree to 2, which adds the x1 times x2 feature, and press Compare again: the same linear model recovers. Feature engineering can replace a more complex model.

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.500 | 0.500 | 0.500 | 221 |
| Decision tree | 0.981 | 0.983 | 0.983 | 22 |
| Random forest | 1.000 | 0.967 | 0.967 | 375 |
| SVM | 0.969 | 0.983 | 0.983 | 6 |
| Neural network | 0.993 | 0.972 | 0.972 | 3129 |

Majority class baseline on the test set: 0.500.

Forest test loss: 0.096 at best, final 0.107 at tree 100.

Follow-up: Linear degree 2

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.974 | 0.983 | 0.983 | 224 |
| Decision tree | 0.981 | 0.983 | 0.983 | 24 |
| Random forest | 1.000 | 0.967 | 0.967 | 412 |
| SVM | 0.969 | 0.983 | 0.983 | 6 |
| Neural network | 0.993 | 0.972 | 0.972 | 3273 |

Majority class baseline on the test set: 0.500.

## 3. Overfitting on small data

Data: `moons`, n = 40, noise = 1.0, minority share = 0.5, test mode = random, seed = 2.

What to look for: k = 1 kNN, the deep tree and the large network reach about 100 percent training accuracy but fall behind on the test set (the Overfitting badge in each tab). Then open the Decision tree tab, set max depth to 2, and compare again: the gap shrinks.

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.833 | 0.938 | 0.938 | 195 |
| kNN | 1.000 | 0.812 | 0.812 | 2 |
| Decision tree | 1.000 | 0.750 | 0.750 | 9 |
| Neural network | 1.000 | 0.562 | 0.562 | 1534 |

Majority class baseline on the test set: 0.500.

Follow-up: Shallow tree

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.833 | 0.938 | 0.938 | 183 |
| kNN | 1.000 | 0.812 | 0.812 | 2 |
| Decision tree | 0.875 | 0.938 | 0.938 | 4 |
| Neural network | 1.000 | 0.562 | 0.562 | 1473 |

Majority class baseline on the test set: 0.500.

## 4. Capacity needed

Data: `spirals`, n = 600, noise = 0.15, minority share = 0.5, test mode = random, seed = 1.

What to look for: The neural network, RBF SVM, random forest and kNN do well. Logistic regression and a depth-2 tree stay far behind.

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.679 | 0.672 | 0.672 | 214 |
| Decision tree | 0.740 | 0.750 | 0.750 | 3 |
| kNN | 1.000 | 1.000 | 1.000 | 4 |
| Random forest | 0.988 | 0.978 | 0.978 | 428 |
| SVM | 1.000 | 1.000 | 1.000 | 10 |
| Neural network | 1.000 | 0.994 | 0.994 | 5554 |

Majority class baseline on the test set: 0.500.

Forest test loss: 0.124 at best, final 0.140 at tree 100.

## 5. The accuracy trap

Data: `blobs_overlap`, n = 1000, noise = 0.8, minority share = 0.05, test mode = random, seed = 1.

What to look for: Compare the accuracy with the majority baseline in each tab, then switch the chart to recall or F1. Open the Linear tab (Edit), set class weight to Balanced, and compare again: recall jumps, and accuracy drops a little. Accuracy alone would have hidden this.

| Model | Train accuracy | Test accuracy | Test recall | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.956 | 0.970 | 0.400 | 220 |
| Decision tree | 0.976 | 0.963 | 0.400 | 32 |
| Random forest | 0.994 | 0.973 | 0.467 | 382 |
| Gradient boosting | 0.981 | 0.963 | 0.467 | 1406 |

Majority class baseline on the test set: 0.950.

Forest test loss: 0.107 at best, final 0.107 at tree 100.

Boosting test loss: minimum 0.095 at tree 19, final value 0.112 at tree 100 (+18 percent).

Follow-up: Linear balanced

| Model | Train accuracy | Test accuracy | Test recall | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.886 | 0.930 | 0.867 | 196 |
| Decision tree | 0.976 | 0.963 | 0.400 | 37 |
| Random forest | 0.994 | 0.973 | 0.467 | 404 |
| Gradient boosting | 0.981 | 0.963 | 0.467 | 207 |

Majority class baseline on the test set: 0.950.

## 6. Product threshold (torque times speed)

Data: `product`, n = 600, noise = 0.2, minority share = 0.5, test mode = random, seed = 1.

What to look for: Logistic regression cannot draw a hyperbola, so it stays low. Trees, forests and boosting approximate the boundary with boxes. Adding the product as a feature (polynomial degree 2) would also fix the linear model.

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.500 | 0.500 | 0.500 | 199 |
| Decision tree | 0.998 | 0.972 | 0.972 | 42 |
| Random forest | 1.000 | 0.978 | 0.978 | 412 |
| Gradient boosting | 1.000 | 0.967 | 0.967 | 614 |

Majority class baseline on the test set: 0.500.

Forest test loss: 0.079 at best, final 0.079 at tree 100.

Boosting test loss: minimum 0.092 at tree 50, final value 0.096 at tree 100 (+5 percent).

Follow-up: Linear degree 2

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.979 | 0.972 | 0.972 | 217 |
| Decision tree | 0.998 | 0.972 | 0.972 | 43 |
| Random forest | 1.000 | 0.978 | 0.978 | 493 |
| Gradient boosting | 1.000 | 0.967 | 0.967 | 825 |

Majority class baseline on the test set: 0.500.

## 7. Regression where linear is enough

Data: `power`, n = 300, noise = 0.4, minority share = 0.5, test mode = random, seed = 1.

What to look for: Linear regression matches or beats the flexible models on test R squared, and it is the fastest and easiest to explain.

| Model | Train r2 | Test r2 | Test r2 | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.854 | 0.884 | 0.884 | 3 |
| kNN | 0.886 | 0.859 | 0.859 | 3 |
| Decision tree | 0.877 | 0.749 | 0.749 | 27 |
| Random forest | 0.969 | 0.833 | 0.833 | 377 |
| Gradient boosting | 0.972 | 0.824 | 0.824 | 185 |
| SVM | 0.857 | 0.859 | 0.859 | 8 |
| Neural network | 0.863 | 0.865 | 0.865 | 1374 |

Forest test loss: 0.118 at best, final 0.119 at tree 100.

Boosting test loss: minimum 0.104 at tree 16, final value 0.125 at tree 100 (+19 percent).

## 8. Regression with an interaction

Data: `inter`, n = 600, noise = 0.2, minority share = 0.5, test mode = random, seed = 1.

What to look for: Linear regression has an R squared near zero, while trees, forests and boosting approximate the saddle. Then open the Linear tab (Edit), set the polynomial degree to 2, and compare again: the engineered x1 times x2 term makes the linear model succeed.

| Model | Train r2 | Test r2 | Test r2 | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.111 | 0.107 | 0.107 | 5 |
| Decision tree | 0.975 | 0.933 | 0.933 | 123 |
| Random forest | 0.982 | 0.949 | 0.949 | 366 |
| Gradient boosting | 0.987 | 0.954 | 0.954 | 178 |
| SVM | 0.971 | 0.971 | 0.971 | 13 |

Forest test loss: 0.025 at best, final 0.030 at tree 100.

Boosting test loss: minimum 0.026 at tree 76, final value 0.027 at tree 100 (+1 percent).

Follow-up: Linear degree 2

| Model | Train r2 | Test r2 | Test r2 | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.973 | 0.977 | 0.977 | 4 |
| Decision tree | 0.975 | 0.933 | 0.933 | 127 |
| Random forest | 0.982 | 0.949 | 0.949 | 378 |
| Gradient boosting | 0.987 | 0.954 | 0.954 | 187 |
| SVM | 0.971 | 0.971 | 0.971 | 16 |

## 9. Extrapolation

Data: `lin`, n = 300, noise = 0.2, minority share = 0.5, test mode = extrapolation, seed = 1.

What to look for: Linear regression keeps following the trend. Trees, forests and boosting go flat beyond the training range, so their test error (RMSE) is several times larger.

| Model | Train r2 | Test r2 | Test rmse | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.980 | 0.663 | 0.129 | 5 |
| Decision tree | 0.984 | -13.014 | 0.831 | 28 |
| Random forest | 0.995 | -14.206 | 0.866 | 304 |
| Gradient boosting | 0.994 | -12.074 | 0.803 | 177 |
| kNN | 0.984 | -12.160 | 0.806 | 2 |

Forest test loss: 0.608 at best, final 0.750 at tree 100.

Boosting test loss: minimum 0.645 at tree 100, final value 0.645 at tree 100 (+0 percent).

## 10. Bagging versus boosting

Data: `lin_outliers`, n = 200, noise = 0.6, minority share = 0.5, test mode = random, seed = 1.

What to look for: Open the Tree curves tab. The boosting test loss reaches a minimum and then rises as it fits the noise, while the random forest curve stays flat and stable.

| Model | Train r2 | Test r2 | Test r2 | Train time (ms) |
|---|---|---|---|---|
| Random forest | 0.946 | 0.666 | 0.666 | 770 |
| Gradient boosting | 1.000 | 0.498 | 0.498 | 198 |

Forest test loss: 0.599 at best, final 0.615 at tree 300.

Boosting test loss: minimum 0.706 at tree 2, final value 0.926 at tree 300 (+31 percent).

## 11. Seeds and data size

Data: `moons`, n = 40, noise = 0.5, minority share = 0.5, test mode = random, seed = 1.

What to look for: The error bars are wide and the ranking of models can flip between seeds. Now set the sample size to Large (3,000) in the data panel and press Compare again: the bars shrink and the ranking becomes reliable.

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.821 | 0.917 | 0.917 (5 seeds: 0.867 +/- 0.067) | 247 |
| kNN | 0.893 | 0.750 | 0.750 (5 seeds: 0.883 +/- 0.113) | 4 |
| Decision tree | 0.964 | 0.917 | 0.917 (5 seeds: 0.883 +/- 0.113) | 10 |
| Random forest | 1.000 | 0.917 | 0.917 (5 seeds: 0.917 +/- 0.053) | 430 |

Majority class baseline on the test set: 0.500.

Forest test loss: 0.246 at best, final 0.272 at tree 100.

Follow-up: Large data

| Model | Train accuracy | Test accuracy | Test accuracy | Train time (ms) |
|---|---|---|---|---|
| Linear / logistic | 0.875 | 0.877 | 0.877 (5 seeds: 0.870 +/- 0.008) | 391 |
| kNN | 0.988 | 0.967 | 0.967 (5 seeds: 0.978 +/- 0.006) | 6 |
| Decision tree | 0.915 | 0.913 | 0.913 (5 seeds: 0.910 +/- 0.007) | 54 |
| Random forest | 0.991 | 0.972 | 0.972 (5 seeds: 0.979 +/- 0.006) | 733 |

Majority class baseline on the test set: 0.500.
