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


## 2. Project Description
In this section we will cover the main aspects of our final project: the data acquisition, the architecture of the neural network model used, and the evaluation.
We will summarize the structure of the code and go into the details of the implementation and the thought process behind the decisions we made along the way.

### 2.1 General Project Structure
First a short overview of the general project file structure:
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

The project consists mainly of a Jupyter Notebook "./final_project.ipynb" that contains everything related to the structure, training and evaluation of the neural network for predicting stock price movement, with some helper functions found in the file "./lib/dataset_prep_functions.py". Another big part however was the acquisition of data itself - everything related to this part of the project is found in the "./lib" folder comprising the "./lib/Data.py" class file and three scripts for gathering data from the internet ("./lib/economic_data_scraper.py", "./lib/financial_data_scraper.py", "./lib/price_data_scraper.py"). All datafiles created while processing the data are saved in the "./datafiles" folder. 
As an aside, we created a requirements.txt file containing all the necessary dependencies to run the code. To install everything activate your conda virtual environment and install pip - then execute "pip install -r requirements.txt" and all dependencies are checked and installed/updated if necessary. Furthermore please make sure that you are connected to the internet when running the code. 

### 2.1 Data Acquisition
The first thing to mention is that we restricted ourselves to S&P500 companies. The S&P500 is an American stock market index based on the market capitalization of 500 large companies (listed either on the NYSE or the NASDAQ stock exchanges), intended to represent the US economy or more specificly the state of the US stock market. Since we decided not to restrict ourselves to predictions based on price development alone, we had to decide on useful features to improve the predictive capacities of our network. 

We categorized all data in this context into 4 different types: 
    - economic data: information about the economic situation of the country the company has its legal seat in (in this case the USA)
            -> we decided on the GDP (gross domestic product) and an unemployment index
            -> another good economic indicator is the S&P500 index but since it is traded on exchanges we put this one under the rubric of price data
    - financial data: information about the financial situation of the company we want to predict the stock price of
            -> here we wanted to represent the three basic part of a companies financial statement: income statement, balance sheet, cashflow statement
            -> we decided to represent the financial statement by the EBIT (earnings before interest and taxes) 
            -> the balace sheet is represented by the total current assets and liabilities
            -> and for the cashflow statement we decided on the net cashflow from operating activities
    - price data: we attributed pretty much everything related to the stocks directly to category price data
            -> the price data consists of the daily closing prices of the given stock, its trading volume of the day and the S&P500 index 
    - sentiment data: this would be data summarizing the current perception of the given company
            -> examples would be data from twitter sentiment analyses or google trends
            -> acquiring this kind of data would go beyond the scope of this project
            -> in principle however it could be added to the current data

In general we were restricted by the availability of most of the data. While price data is easily accessable in most cases, gathering and processing economic and financial data proved to be quiet challenging. The general problem with economic data was that all the data was only available for quarterly periods and we decided on a simple linear interpolation to extend the data points to daily values. We faced the same problem with the financial data and solved it similarly, but in addition we had difficulties to get access to historical financial data (more than last 2 years) of companies in general. Because there do not exist useful APIs for accessing historical financial data for larger time intervals (for free), we had to develop code to scrape the data from the html code of a website (https://ih.advfn.com/stock-market/NASDAQ) that contained the information for most S&P500 companies. Furthermore there are probably way more useful financial indicators like the EPS (earnings per share), the P/E (the price/earnings ratio) or the P/B (price/book-value ratio), but it is even harder to get consistent information about those values and so we decided on more fundamental indicators that are easier accessable.

The Data.py class file of the project contains our solution to handling the problems we faced and to get all the data we decided upon. 
To summarize it briefly: it contains a Data class which instatiates a Data object for a given stock with a given start date. 
In this instantiation all the requested data is gathered - either it is already available in the datafiles folder or gets downloaded from the internet via the previously mentioned scripts ("./lib/economic_data_scraper.py", "./lib/financial_data_scraper.py", "./lib/price_data_scraper.py").
As an example, Data('AAPL', '2000-01-01') instantiates a Data object for the Apple stock, containing the data from January 1st 2000 upto today. 
More implementation details are found in the documentation of the class.

Obviously we are restricted to the data available on the website and from the APIs used for price (alpha_vantage) and economic data (quandl), so some companies are note available and the range of data may also be limited in some cases. 

We decided to test everything mainly for the company American Airlines (AAL) because it exhibits strong price fluctuations without showcasing an obvious trend. 

### 2.2 Data Preparation


### 2.3 Model Architecture and Training


### 2.4 Evaluation


## 3. Result

