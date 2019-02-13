# Stock Market Prediction Using LSTM

## 1. Introduction

### 1.1. Stock Prediction Using Neural Networks

The most valuable piece of information in the stock market is an estimation of the share prices or their growth direction in the future. Efforts for predict the stock market behavior are as old as the market itself. These include a wide range of efforts from purely economic and technical analyses to the statistical data mining, to the ML-aided methods like decision trees.

No need to mention that none of these efforts have ever been, practically, successful. In fact, there are reputable speculations, including 'Random Walk' hypothesis or 'Efficient Market' hypothesis that state the market behavior is essentially unpredictable. Even though, it is not a surprise that researchers are still working on the stock prices prediction problem.

With the hype of the artificial neural networks and their mysterious success in approximating the most complex functions and the most nonlinear chaotic relations, hopes for market behavior prediction were raised. The advantage of neural network seems to be the fact that they can work without the explicit knowledge of the operator. That is, a neural network is able to practically learn unformalized relations that are not, explicitly, known to us.

Taking neural gates into account, it is almost three decades that artificial neural architectures are being employed as a tool for predicting stock prices. These studies could be categorized based on answer to three questions:

1. **What are the input fed into the network?**
There are numerous different combinations of factors that were tried as the network inputs. Prices history is obviously the essential factor; many studies successfully used it as the only input. But the results generally show improvement with adding more factors that cover various dimensions of the companies financial status, like assets, liabilities, ..., the economic environment like GDP, inflation rate, ..., and as a recent trend, the sentiment and textual data mined from social media like Twitter or even Google Trends.
The main challenge is to find a perfect set of inputs. First and the most basic consideration is that the input set should capture enough relevant data necessary for reconstructing the output value. So the set should be comprehensive. However, some studies show that feeding too many weakly correlated input factors reduces the accuracy of prediction by confusing the network on irrelevant patterns. So the input set should also be well-restricted.
The more general formulation of the problem is to find a balance between trust to our humanly-achieved knowledge of economy dynamics and the trust to poorly-realized ways in which a neural network discover patterns in data.

2. **What is the class/architecture of the network?**
Different classes of neural network have been employed for market behavior prediction, including a simple MLP to some unprecedented combinations of multilayer stacked LSTM. And within each class a network could vary due to its especially designed architecture, through changing hyperparameters like the number of hidden layers or hidden nodes and/or due to the activation function(s) it uses.

3. **What does the network predict?**
The networks usually meet their goal if they could tell us whether the prices fall, remain static, or rise. But having a real value for the prices in a near future is also a popular goal.

### 1.2. Experiment Design

**Input**

For the experiment, we decided to use three out of four categories of input data: historical, financial, and environmental.
There are possibilities for refining data for a better result, like using P/E (price-to-earnings ratio) and P/B factors in (price-to-book value ratio), but due to competitive reason companies are inclined not to disclose all their financial data. Therefore, we decided to keep working with the best we could have access free and online.

**Model**

For the neural network, we chose to work with an LSTM. In general, RNNs are designed to work with sequential data. That includes, one-to-many, many-to-one, and many-to-many relations between inputs and outputs. In our experiments, sequence of historical data are feed into the network in order to get one single piece of data, as prediction, back.
LSTMs (Long Short-Term Memory) are a specialized and the most successful variation of RNNs, due to their ability to tackle the training problem of classic RNNS: the long-term dependencies. On the other hand, the price prediction task demands an essential capability for learning patterns over a long period of time. Therefore, we decided to use an LSTM for the experiment.

**Output**

From a ML point of view, there are minor differences between a network that can advise one to sell, hold, or buy a specific company's shares and a network that predicts the stock price for the same company for a specific date in future. Both should be able to learn market behavior; one within a classification and the other within a regression task.
The main difference between these two kinds of network is that designing the first one needs some intuition understanding of economic dynamics. Accordingly, we decided to work with net prices and remove the analytic part of the task.


## 2. Experiment Description

### 2.1. Data Acquisition

### 2.2. Model Architecture

### 2.3. Evaluation


## 3. Result

### 3.1. Achievements

### 3.2. Limitations
- limitations like data privacy for financial security reasons, company's financial statement
