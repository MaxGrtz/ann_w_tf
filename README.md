# Stock market prediction with a Recurrent Neural Network

## 1. Introduction to Stock Prediction
The most valuable piece of information in the stock market is an estimation of the share prices or their growth direction in the future. Efforts for predict the stock market behavior are as old as the market itself. These include a wide range of efforts from purely economic and technical analyses to the statistical data mining, to the ML-aided methods like decision trees.

No need to mention that none of these efforts have ever been, practically, successful. In fact, there are reputable speculations, including 'Random Walk' hypothesis or 'Efficient Market' hypothesis that state the market behavior is essentially unpredictable. Even though, it is not a surprise that researchers are still working on the stock prices prediction problem.

With the hype of the artificial neural networks and their mysterious success in approximating the most complex functions and the most nonlinear chaotic relations, hopes for market behavior prediction were raised. The advantage of neural network seems to be the fact that they can work without the explicit knowledge of the operator. That is, a neural network is able to practically learn unformalized relations that are not, explicitly, known to us.

Taking neural gates into account, it is almost three decades that artificial neural architectures are being employed as a tool for predicting stock prices. These studies could be categorized based on answer to three questions:

1. **What are the input fed into the network?**
There are numerous different combinations of factors that were tried as the network inputs. Prices history is obviously the essential factor; many studies successfully used it as the only input. But the results generally show improvement with adding more factors that cover various dimensions of the companies financial status, like assets, liabilities, ..., the economic environment like GDP, inflation rate, ..., and as a recent trend, the sentiment and textual data mined from social media like Twitter or even Google Trends.
The main challenge is to find a perfect set of inputs. First and the most basic consideration is that the input set should capture enough relevant data necessary for reconstructing the output value. So the set should be comprehensive. However, some studies show that feeding too many weakly correlated input factors reduces the accuracy of prediction by confusing the network on irrelevant patterns. So the input set should also be well-restricted.
The more general formulation of the problem is to find a balance between trust to our humanly-achieved knowledge of economy dynamics and the trust to poorly-realized ways in which a neural network discover patterns in data.

2. **What is the type/architecture of the network?**
Many different types of neural network were employed for market behavior prediction, including a simple MLP to some unprecedented combinations of multilayer stacked LSTM. And each type could use a different architecture, using its own hyperparameters like the number of hidden layers or hidden nodes and/or due to the activation function(s) it uses.

3. **What does the network predict?**
The networks usually meet their goal if they could tell us whether the prices fall, remain static, or rise. But having a real value for the prices in a near future is also a popular goal.


## 2. Recurrent Neural Networks for Stock Prediction
- why we recurrent neural nets for time series prediction?
- why lstm?


## 3. Description of the experiment

### 3.1 Data Acquisition

### 3.2 Model Architecture

### 3.3 Evaluation


## 4. Result

### 4.1 Achievements

### 4.2 Limitations
- limitations like data privacy for financial security reasons, company's financial statement
