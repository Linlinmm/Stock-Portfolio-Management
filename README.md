# Stock-Portfolio-Management
Implementation of Paper：Contrastive Hypergraph Attention Reinforcement Learning for Stock Portfolio Management
## Abstract
Stock portfolio management (SPM) is a task of continuously re-allocating an amount of funds to a set of stock assets for the purpose of maximizing investment rewards or suppressing risks. Recent SPM methods learn a policy function to generate appropriate portfolios in a reinforcement learning framework and achieve encouraging progress.
However, these methods ignore the fact stock assets in portfolios may be broadly interrelated and affected by each other,and the stock relationship information is highly value in improving portfolio management. In this paper, we introduce the group-wise relationship information as additional environmental cues to improve portfolio policy learning for SPM. We propose a dual channel hypergraph attention network to jointly capture the heterogeneous industry-belonging and fund-holding relationships via the attention based information aggregation among stocks. In addition, contrastive learning is introduced to provide extra supervision signals for higher modeling capacity. we conduct investment simulation on the real-world dataset collected from China’s A-share market. The experimental results demonstrate the significant superiority of our method compared to the state-of-the-art methods for SPM, in terms of both accumulated profits and risk-adjusted profits.
## Dataset
In our experiments, we crawled the historical prices of stocks in China’s A-share market via a financial data API https://tushare.pro/document/2. We chose the stocks having price records between 01/04/2013 and 12/31/2019, and further filtered out those stocks traded on less than 98% of all trading days, finally leading to 758 stocks. The price data were chronologically split into three time periods in an approximate ratio of 8:1:1 for training],validation ,and testing,respectively.  
we grouped all stocks into 104 industry categories according to the Shenwan Industry Classification Standard https://www.swsresearch.com/institute_sw/allIndex/releasedIndex; for the latter,we selected 61 mutual funds established before 2013 in the A-share market, and acquired the constituent stocks of each fund from the quarterly portfolio reports.
## Models
`/training/models.py`: the policy network framework  
`/trainin/module.py`: implementation of dual channel hypergraph attention module  
`/training/batch_loss.py`: Calculate cumulative reporting and related revenue metrics  
`/training/train.py`: train this project  
