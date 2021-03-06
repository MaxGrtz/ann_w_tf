# Stock market prediction with a Recurrent Neural Network

## 1. Introduction

### 1.1. Stock Prediction Using Neural Networks
The most valuable piece of information in the stock market is an estimation of the share prices or their growth direction in the future. Efforts for predict the stock market behavior are as old as the market itself. These include a wide range of efforts from purely economic and technical analyses to statistical data mining, to the ML-aided methods like decision trees.

No need to mention that none of these efforts have ever been, practically, successful. In fact, there are reputable speculations, including 'Random Walk' hypothesis or 'Efficient Market' hypothesis that state the market behavior is essentially unpredictable. Even though, it is not a surprise that researchers are still working on the stock prices prediction problem.

With the hype of the artificial neural networks and their apparent success in approximating highly complex functions and nonlinear chaotic relations, hopes for market behavior prediction were raised. The advantage of neural network seems to be the fact that they can work without the explicit knowledge of the operator. That is, a neural network is able to practically learn unformalized relations that are not explicitly known to us.

Taking Perceptrons into account, it is almost three decades that artificial neural architectures are being employed as a tool for predicting stock prices. These studies could be categorized based on answers to three questions:

**1. What are the inputs fed into the network?**
There are numerous different combinations of factors that were tried as network inputs. Price histories are obviously the essential factor; many studies successfully used it as the only input. But the results generally show improvement with adding more factors that cover various dimensions of the companies financial status, like assets, liabilities, ..., the economic environment like GDP, inflation rate, ..., and as a recent trend, the sentiment and textual data mined from social media like Twitter or even Google Trends.
The main challenge is to find a perfect set of inputs. First and the most basic consideration is, that the input set should capture enough relevant data necessary for reconstructing the output value. So the set should be comprehensive. However, some studies show that feeding too many weakly correlated input factors reduces the accuracy of prediction by confusing the network on irrelevant patterns. So the input set should also be well-restricted.
The more general formulation of the problem is to find a balance between trust in our humanly-achieved knowledge of economy dynamics and the trust in the abilities of a neural network to discover patterns in data that seems uninformative to us.

**2. What is the class/architecture of the network?**
Different classes of neural network have been employed for market behavior prediction, including simple Multi-Layer Perceptrons (MLPs) to some unprecedented combinations of multilayer stacked LSTM. And within each class a network could vary due to its specific configuration architecture, through changing hyperparameters like the number of hidden layers or hidden nodes and/or due to the activation function(s) it uses.

**3. What does the network predict?**
There are two main approaches to predictions with respect to the stock market. The first is to predict trading decisions, making it a classification task, deciding wheather to buy or sell a stock, which is eventually based on a prediction of rising of falling prices. The second is a regression approach, explicitly predicting stock price movement over some horizon into the future. 


### 1.2. Experiment Design

**Input**
For the experiment, we decided to use three categories of input data: historical, financial, and economic data.

**Model**
For the neural network, we chose to work with an LSTM. In general, RNNs are designed to work with sequential data. That includes, one-to-many, many-to-one, and many-to-many relations between inputs and outputs. In our experiments a sequence of historical data is fed into the network in order to get a prediction of the price of the successive day. 
LSTMs (Long Short-Term Memory) are a specialized and the most successful variation of RNNs, due to their ability to tackle a major problem of classic RNNS: learning long-term dependencies.

**Output**
From a Machine Learning point of view, there are minor differences between a network that can advise one to sell, hold, or buy a specific company's shares and a network that predicts the stock price for the same company for a specific date in future. Both should be able to learn market behavior; one within a classification and the other within a regression framework. We decided on predicting stock prices explicitly because their performance is easier to evaluate.  


## 2. Project Description
In this section we will cover the main aspects of our final project: the data acquisition, the architecture of the neural network model used, and the evaluation.
We will summarize the structure of the code and go into the details of the implementation and the thought process behind the decisions we made along the way.

### 2.1 Goal of the Project
The goal of the project, in general, was to develop a deeper understanding of recurrent neural networks (RNNs) by applying them to the classical problem of stock price prediction. Stock prices are the prototypical example of sequential/time-series data and RNNs, specifically RNNs with Long Short-Term Memory (LSTM) cells, are the state of the art in Deep Learning  when it comes to learning and predicting sequencial patterns.
We used the Tensorflow framework to build a RNN with LSTM cells and applied it to stock prices and other company related data to predict future price development.
In the following paragraphs we go into the details of the different aspects of the project. 

### 2.2 General Project Structure
First a short overview of the project files structure:

```bash
.
├── Final_Task_Info_Sheet.pdf
├── README.md
├── datafiles
├── final_project.ipynb
├── lib
│   ├── Data.py
│   ├── __init__.py
│   ├── dataset_prep_functions.py
│   ├── economic_data_scraper.py
│   ├── financial_data_scraper.py
│   └── price_data_scraper.py
└── requirements.txt
```

The project mainly consists of a Jupyter Notebook "./final_project.ipynb" that contains everything related to the structure, training and evaluation of the neural network for predicting stock price movement, with some helper functions found in the file "./lib/dataset_prep_functions.py". Another main part, however, was the acquisition of data itself - everything related to this part of the project is found in the "./lib" folder comprising the "./lib/Data.py" class file and three scripts for gathering data from the internet ("./lib/economic_data_scraper.py", "./lib/financial_data_scraper.py", "./lib/price_data_scraper.py"). All datafiles created while processing the data are saved in the "./datafiles" folder. 
As an aside, we created a requirements.txt file containing all the necessary dependencies to run the code. To install everything activate your conda virtual environment with python 3.7x and install pip - then execute "pip install -r requirements.txt" and all dependencies are checked and installed/updated if necessary. Furthermore please make sure that you are connected to the internet when running the code. 


### 2.3 Data Acquisition
The first thing to mention is that we restricted ourselves to S&P500 companies. The S&P500 is an American stock market index based on the market capitalization of 500 large companies (listed either on the NYSE or the NASDAQ stock exchanges), intended to represent the US economy or more specifically the state of the US stock market. Since we decided not to restrict ourselves to predictions based on price development alone, we had to decide on useful features to improve the predictive capacities of our network. 

We categorized all data in this context into 4 different types:

**1. economic data:** information about the economic situation of the country the company has its legal seat in (in this case the USA)

 - we decided on the GDP (gross domestic product) and an unemployment index
 - another good economic indicator is the S&P500 index but since it is traded on exchanges we put this one under the rubric of price data


**2. financial data:** information about the financial situation of the company we want to predict the stock price of

 - here we wanted to represent the three basic parts of a companies financial statement: income statement, balance sheet, cashflow statement
 - we decided to represent the financial statement by the EBIT (earnings before interest and taxes) 
 - the balance sheet is represented by the total current assets and liabilities
 - and for the cashflow statement we decided on the net cashflow from operating activities


**3. price data:** we attributed pretty much everything related to the stocks directly to the category price data

 - the price data consists of the daily closing prices of the given stock, its trading volume of the day and the S&P500 index 


**4. sentiment data:** this would be data summarizing the current perception of the given company

 - examples would be data from twitter sentiment analyses or google trends
 - acquiring this kind of data would go beyond the scope of this project
 - in principle however it could be added to the current data

In general we were restricted by the availability of most of the data. While price data is easily accessable in most cases, gathering and processing economic and financial data proved to be quiet challenging. The general problem with economic data was that all the data was only available for quarterly periods and we decided on a simple linear interpolation to extend the data points to daily values. We faced the same problem with the financial data and solved it similarly, but in addition we had difficulties to get access to historical financial data (more than last 2 years) of companies in general. Because there do not exist useful APIs for accessing historical financial data for larger time intervals (for free), we had to develop code to scrape the data from the html code of a website (https://ih.advfn.com/stock-market/NASDAQ) that contained the information for most S&P500 companies. Furthermore there are probably financial indicators that are better suited for predictions like the EPS (earnings per share), the P/E (the price/earnings ratio) or the P/B (price/book-value ratio), but it is even harder to get consistent information about those values and so we decided on more fundamental indicators that are easier accessible.

The Data.py class file of the project contains our solution to handling the problems we faced and to get all the data we decided upon. 
To summarize it briefly: it contains a Data class which instatiates a Data object for a given stock with a given start date. 
In this instantiation all the requested data is gathered - either it is already available in the data files folder or gets downloaded from the internet via the previously mentioned scripts ("./lib/economic_data_scraper.py", "./lib/financial_data_scraper.py", "./lib/price_data_scraper.py").
As an example, Data('AAPL', '2000-01-01') instantiates a Data object for the Apple stock, containing the data from January 1st 2000 up to today as a pandas dataframe. More implementation details are found in the documentation of the class.

Obviously we are restricted to the data available on the website and from the APIs used for price (alpha_vantage) and economic data (quandl), so some companies are not available and the range of data may also be limited in some cases. 


### 2.4 Data Preparation
After gathering all the necessary information about a stock, the next step was to prepair the data for the training process and predictions with the RNN. 
There are essentially three parts to this preparation: creating target labels, splitting of the data into subsequences, and normalization.

**1. Creating Target Labels**

As target labels for the predictions of our RNN we did choose the features of the following day. It was necessary to predict all features of the following day and not only the price because in the actual testing phase we wanted to propagate predictions into the future up to the prediction horizon we decided on, so the output of our network actually had to be a valid input to the network as well.

**2. Splitting of the Data into Subsequences**

As usual the Truncated Backpropagation through Time (TBTT) algorithm is used for training the RNN, which required to split the complete timeseries of the feature data and corresponding labels into subsequences of a defined length. To increase the number of available training examples we chose to create overlapping subsequences (shifted by one day). Furthermore we split the list of those subsequences into training and validation data, were about 10% of the subsequences closest to the current date are used as validation data. 
Additionally we created non overlapping subsequences (of the same length) as inputs for the final predictions.

**3. Normalization**

For normalization it seemed to be optimal to normalize every subsequence individually. We used the following formula to normalize the i'th datapoint in a subsequence: n_i = (p_i/p_0) - 1 such that p_0 has the value 0 and the following p_i are measured relative to the first datapoint of a sequence.
The training and validation data could be normalized in advance, whereas the prediction data had to be normalized during the prediction process since the p_0 values of each subsequence had to be stored to invert the normalization for the RNN outputs with the formula: p_i = (n_i + 1) * p_0.


### 2.5 Model Architecture and Training
As already mentioned the RNN was implemented using the Tensorflow framework and we designed the network such that most essential parameters defining the network structure and the training process can be adjusted under the "Training and Model Configurations" section of the Jupyter Notebook. 
Those parameters are: Validation Data Ratio, Subsequence Length, Prediction Horizon, LSTM Sizes, Batch Size, Learning Rate and Number of Epochs.

 - Validation Data Ratio: defines as a decimal the percentage of data we want to use as validation data

 - Subsequence Length: as already mentioned defined the length of data sequences (for the TBTT algorithm)

 - Prediction Horizon: defines how many days we want to predict into the future in the final predictions

 - LSTM Sizes: is a list of integers defining the number of nodes in each LSTM layer 
   (hidden state and cell state therefore having the same number of nodes)

 - Batch Size: number of subsequences used for a single training step ie. weight update 
   (averaging over the loss of the individual sequences)

 - Learning Rate: controlling the size of the update step in parameter space

 - Number of Epochs: defines how many times the training data is passed through for training

**1. Architecture - Graph definition**

First we defined placeholder for the input batch (one for feature input and one for target labels) and for the hidden and cell state of the LSTM layers. The RNN itself consists of tf.nn.rnn_cell.LSTMCell objects for each LSTM layer with the state sizes defined in the parameter LSTM Sizes. As activation function for each Cell we did chose tanh, since the output domain [-1,1] matches the normalized prediction, which are based on percentages. Each LSTM Cell, representing a Layer of the RNN, is wrapped in a Dropout layer with a dropout of 5% (only for training phase as indicated by a training-flag boolean placeholder) for regularization purposes (to avoid overfitting to the training data). 
The wrapped LSTM cells are than stacked using the tf.nn.rnn_cell.MultiRNNCell API, creating the complete RNN. Afterwards the computations are defined, where the tf.nn.dynamic_rnn is used to run the calculations for the input batch, given the current RNN state. The outputs are the corresponding predictions for every sequence of the input batch and the final state of the RNN. 
The prediction output of the RNN is then mapped to the correct output dimensions by a final linear dense layer. 
The loss is calculated as the mean squared error (MSE) between the output of the dense layer and the target values. To minimize the loss we decided on the ADAM Optimizer, with the defined Learning rate parameter as starting learning rate. 

**2. Training, Validation and Prediction**

The training consisted of the defined number of epochs. Each epoch the training data was shuffled and split according to the defined batch size. The RNN state was set to all zeros at the beginning of every epoch as well. One Training step was performed for every batch and the final state of the RNN after every training step was propagated to the next. 
For the validation data we wanted to check every sequence individually (and not cut off any by splitting them into batches), so we duplicated the every single sequence (depending on the batch size) such that it had the correct dimensions for the batch placeholder of the RNN. The same manipulation was applied to the test data to accommodate for the fact that we wanted to get predictions for every sequence individually. 
The loss for training and validation are written to the train and validation files in the summary folder and can be visualized using tensorboard.

As mentioned earlier, the test data sequences were normalized by the same formula used for normalizing the training and validation data, but the first value of every sequence had to be saved to invert the normalization afterwards. 
For every prediction the duplicated input sequence in fed into the RNN and the output is a prediction of the same sequence length, shifted one day into the future.
This last value of the output sequence is the value the network predicts for the day following the end of the input sequence. This value is appended to the input sequence, while the first value is cut off. This process is repeated based on the number of days we want to predict into the future. The predictions for the following n days (only the stock price value, not the predictions for all other features) are then saved in a list of predictions after inverting the normalization. The process in repeated for every subsequence in the test data.

### 2.6 Visualization and Evaluation
For visualization purposes the predictions saved in the list of predictions are plotted against the true stock prices. This gives a good insight into how well the LSTM captures the patterns of the price sequence and how well it predicted the data it was never trained on (how well it generalizes). Additionally we calculated the Root Mean Squared Error (RMSE) between the final predictions and the true prices as a measure of the average prediction error. We did chose the RMSE, because it is easily interpretable since it actually has the same units as the values compared, ie. dollars. We considered using the root of the sum squared error to get some absolute measure of how well our predictions are over the whole timespan, but this proved to be problematic when comparing different network configurations. The reason is that the number of test sequences in the current implementation depends on the length of subsequences and therefore differs across configurations. Consequently a mean value like the RMSE seemed a better choice for the following evaluation. 

For the final evaluation of our model we decided to use the company American Airlines (AAL), because it exhibits strong price fluctuations without showcasing any obvious trends. 

We tested varying configurations to find some good parameters combination.
Generally we decided to fix the prediction horizon to about a month ie. 30 days. We created an overview over the RMSE results of different configurations which is attached to the jupyter notebook. It shows that we tested for two different Sequence lenghts: 50 and 100 days. For each sequence length we decided on a constant batchsize of 10 and only tried different learning rates (1.00E-06, 1.00E-05, 5.00E-05, 1.00E-04), because increasing the batchsize is somewhat equivalent/has comparable effects to decreasing the learning rate. Three network architectures ([128,64], [128,128], [128,128,128]) were compared for different numbers of epochs (50, 100,150,200).  

As an example for the learning behavior of our implementation here is one graph visualizing the training and validation loss for the following configuration: 


```python
# configs
SUBSEQ_LEN = 100  # length of sequences fed into rnn
HORIZON = 30  # number of days to predict into future
VALID_RATIO = 0.1  # percent of validation data
BATCH_SIZE = 10
LSTM_SIZES = [128, 64]  # size of lstm states per layer
EPOCHS = 100
LEARNING_RATE = 1e-5
```

<p align="center">
  <img width="400" height="300" src="./datafiles/training_loss.png">
</p>

<p align="center">
  <img width="400" height="300" src="./datafiles/validation_loss.png">
</p>


For the same run, here is a visualization of the final predictions:

<p align="center">
  <img width="900" height="500" src="./datafiles/predictions.png">
</p>

The final RMSE of this run was: 2.4937


Finally we compiled an overview of the results, comparing hyperparameter configurations by the RMSE of the final test predictions. 

<p align="center">
  <img width="1200" height="800" src="./datafiles/results_overview.png">
</p>



The evaluation showed, that longer sequence lengths (100 days), as expected, showed better overall prediction performance. 
Further there does not seem to be any obvious advantage to choose more complex network structures, since the three layer network did not perform any better on the runs we did, than the simpler networks. It should however be mentioned that a single run for every configuration does not provide sufficient data for a comprehensive analysis.


### 3. Further Ideas
There are two major aspect that we would have liked to explore more, but could not because it would have gone beyond the scope of this project. The first is a more sophisticated choice of input features, which was (as mentioned above) constrained by the availability of data, and an extension to sentiment indicators such as google trends or twitter analyses. This kind of data could provide evidence of interest in a company and the mainstream perception of it and thus prove to be a useful predictor of price development.
A second idea that would have been interesting to explore, is to train a network on the data of multiple companies to extract more general patterns.