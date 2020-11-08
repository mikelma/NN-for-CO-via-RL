# Experimentation Results

## 2020-10-07
(commit: `387333aaae049d430bc800a88731e6e70dd39359`)

### Hyperparameters:
- *lr*: .0003
- *num. samples*: 64
- *iterations*: 2000
- *noise vec. length*: 128

### The model:
Feed-forward neural network (classic MLP):

| Layer | Type                | Num. Nodes              | Activation func. |
| ----- | ------------------- | ----------------------- | ---------------- |
| 1.    | Input / Shared      | 512                     | ReLU             |
| 2.    | Output (ModuleList) | [N-i for i in range(N)] | None             |

### Experment setup
For this experiment the used utility function was: 

$$U(v_i) = f(v_i) - \frac{1}{m}\sum_{j=1}^m {f(v_j)}$$

The problem was *PFSP* and the instance `tai_20_5_8`.

### Results

 ![Minimun fitness and loss value over iterations](img/2020_11_07_1.png){ width=100% }


Metrics of the last iteration:
- *best fitness*: $15095$
- *min fitness*:  $15107$
- *mean fitness*: $15107$

This experiment was just a demo to demonstrate that our approach works. However, it turned out that the obtained results are quite competitive compared to other NN approaches (model's parameters such as num. of layers nd nodes were taken arbitrarily), taking into account that state of the art meta-heuristics obtain mfitness values around $1.44\times10^4$ and $1.47\times10^4$.
