# Experimentation Results

**Important note**: Until section \textit{Experiments with model's convergency} there was an error in the code, and only $3$ solutions were sampeld from the model in each iteration (instead of the number of samples defined in the experiment setup). This caused the expected fitness value (see the loss function definition below) to be inacurate and increasing the stochasticity of the optimization process. This is a minor issue for the first experiments as their only purpose is to demonstrate that the neural network apprach works.

## Demo 1
(commit: `387333aaae049d430bc800a88731e6e70dd39359`)

### The model:
Feed-forward neural network (classic MLP):

| Layer | Type                | Num. Nodes              | Activation func. |
| ----- | ------------------- | ----------------------- | ---------------- |
| 1.    | Input / Shared      | 512                     | ReLU             |
| 2.    | Output (ModuleList) | [N-i for i in range(N)] | None             |

### Experment setup

* *lr*: .0003
* *num. samples*: 64
* *iterations*: 2000
* *noise vec. length*: 128

For this experiment the used utility function was: 

$$U(v_i) = f(v_i) - \frac{1}{m}\sum_{j=1}^m {f(v_j)}$$

The problem was *PFSP* and the instance `tai_20_5_8`.

### Results

![Minimun fitness and loss value over iterations](img/2020_11_07_1.png){ width=90% }


Metrics of the last iteration:

* *best fitness*: $15095$
* *min fitness*:  $15107$
* *mean fitness*: $15107$

This experiment was just a demo to demonstrate that our approach works. However, it turned out that the obtained results were quite competitive compared to other NN experiments (model's parameters such as num. of layers nd nodes were taken arbitrarily), taking into account that state of the art meta-heuristics obtain fitness values around $1.44\times10^4$ and $1.47\times10^4$.


As can bes seen in the image, the minimun and mean fitness are practically the same from iteration 700 until the end. This suggests that the entropy of the model drops to very low values to early (premature convergence of the model).
To confirm the hipothesis, the same experiment was run, but this time collecting the entropy of the distribution the model outputs. This was the obtained graph:

 ![Same experiment with entropy metric](img/2020_11_08_1.png){ width=90% }
 
At this point, the hipothesis was confirmed, the entropy of the model drops to almost $0$ very rapidly, thus the model converges and no exploration is done.


## More to explore! 

After the premature convergence problem was discorvered, I started thinking solutions to encourage the model to explore more. Finally the following loss function was formulated,

$$L(\theta) = L_1(\theta) - c L_2(\theta)$$
$$L_1(\theta) = \mathbb{E}_{v\sim\theta}[logP(v|x, \theta) U(v)]$$
$$L_2(\theta) = \sum_{i=1}^{N} H(v_i|x,\theta)$$

Where, $\theta$ are the parameters of the NN, $x$ is the random noise vector, $U$ is the utility function and $c$ is a constant parameter (new hyperparameter) that determiness the importance or weight of the $L_2$ loss.

### Experment setup
Hyperparametrs remain unchanged from the last experiment, but the new $c$ parameter is introduced:

* *lr*: $0.0003$
* *num. samples*: $64$
* *iterations*: $2000$
* *noise vec. length*: $128$
* $c$: $40$

For this experiment the used utility function was: 

$$U(v_i) = f(v_i) - \frac{1}{m}\sum_{j=1}^m {f(v_j)}$$

The problem was *PFSP* and the instance `tai_20_5_8`.

###  Results
 
![Training metrics with the new loss function\label{2020_11_08_2}](img/2020_11_08_2.png){ width=90% }

Results are presented in Figure \ref{2020_11_08_2}. This are the fitness values in the last iteration:

* mean fitness: $15442.66$
* min fitness: $15204$
* best fitness: $14704$

The fitness values are very different from each other (in the last experiment this was no the case), suggesting that in this case the sampled solutions are more diverse. Also, looking to the results graph, the entropy value is much higher when using the new loss function. 

In order to better understand the behavior of the model under this new loss function, the same experiment was executed again, but in this case the number of iterations was set to $400$, see Figure \ref{2020_11_08_3}.
 
![Training metrics with the new loss function and 4000 iterations.\label{2020_11_08_3}](img/2020_11_08_3.png){ width=90% }
 
This are the fitness values in the last iteration:

* mean fitness: $15424$
* min fitness: $15305$
* best fitness: $14656$

The ad-hoc implementation of an UMDA operating under the permutation space 
reaches $1.47\times10^4$ as the best fitness value in this instance of the PFSP. **In this experiment the model outperformed the UMDA obtaining $1.46\times10^4$ as the best fitness value!**

# Experiments with model's convergency

In order to better understand the evolution of the probability distribution the model outputs over execution's iterations, distribution's entropy and the probability of the best and worst solution were recorded in each iteration. 
In the following experiments, the value of the $C$ parameter is changed in order to see how the entropy term of the new loss function affects the model. The experimental setup of this experiments is the same as in the previous section (see Section \textit{More to explore!}) 

In Figure \ref{2020_11_15_2}, training metrics are shown for single execution. In this execution the parameter $C$ was set to $0$, thus canceling the entropy term of the new loss function.

![Training metrics of a single execution, where $C=0$. \label{2020_11_15_2}](img/2020_11_15_2.png){ width=110% }

Figure \ref{2020_11_15_2} clearly shows that the model suffers for premature convergency.
Starting from the top left plot, where the fitness of the sampled solutions are presented, from iteration $900$ onwards, minimum and mean fitnesses are equal, suggesting that all (or the majority) of the sampled solutions are identical.
Continuing with the next plot (top right plot in Figure \ref{2020_11_15_2}), the loss function converges to $0$ when the minimum and mean fitness values converge. This is logical as the utility function $U(v)$ tends to $0$ when the fitness of the sampled solution is similar to the mean fitness of the sampled solutions.

Finally, the bottom plot in Figure \ref{2020_11_15_2} shows the convergency metrics of the model over the execution iterations. As can be seen, the entropy of the model rapidly drops to $0$ (note that the maximum entropy the model could have considering $n=20$, is $42.335$) in about $700$ iterations. 

If we take a look to the probability of the best sampled solution, the probability rapidly increases from almost $0$ to a $1$, and remains with the highest probability for the whole execution.
In the other hand, the probability to sample the worst sampled solution aggressively oscillates between $0$ and $1$. The probability fluctuates more in the begining of the execution and slowly becomes more and more stable. This suggests that the entropy of the model is not exactly $0$, but a value proximal to $0$, letting some small probability to sample different solutions. In other words the convergency of the model is not absolute, as there is some small probability to sample new solutions.

In order to evaluate the model's convergency with respect to the $C$ parameter (remind that $C$ is the parameter that weight's the entropy term in the loss function).
The algorithm (with the same experimental setup as the latter experiment) was executed $3$ times for each $C={0, 1, 40}$. The obtained results are shown in Figure \ref{2020_11_15_1}.

![$C$ parameter value comparison. Note that the values above $1$ in the bottom left plot are the standard deviation of the probability, not the probability value itself.\label{2020_11_15_1}](img/2020_11_15_1.png){ width=110% }
