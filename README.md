# Stock-Portfolio-Management
Implementation of Paper：Contrastive Hypergraph Attention Reinforcement Learning for Stock Portfolio Management
## Stock-Portfolio-Management
A short abstract of up to 300 words written in one paragraph, clearly indicating the object and scope of the paper as well as the results achieved, should appear on the first page. It should be written using the abstract environment.
## Dataset
In our experiments, we crawled the historical prices of stocks in China’s A-share market via a financial data API https://tushare.pro/document/2. We chose the stocks having price records between 01/04/2013 and 12/31/2019, and further filtered out those stocks traded on less than 98% of all trading days, finally leading to 758 stocks. The price data were chronologically split into three time periods in an approximate ratio of 8:1:1 for training],validation ,and testing,respectively.  
we grouped all stocks into 104 industry categories according to the Shenwan Industry Classification Standard https://www.swsresearch.com/institute_sw/allIndex/releasedIndex; for the latter,we selected 61 mutual funds established before 2013 in the A-share market, and acquired the constituent stocks of each fund from the quarterly portfolio reports.
## Models
`your content`/training/models.py: the policy network framework  
/trainin/module.py: implementation of dual channel hypergraph attention module  
/training/batch_loss.py: Calculate cumulative reporting and related revenue metrics  
/training/train.py: train this project  
