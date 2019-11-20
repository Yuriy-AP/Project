Yuriy Podvysotskiy
Updated: November 20, 2019
Web-app: https://yuriyp.shinyapps.io/investing_app/

## Project:
---------------------------------------------------

### Plot 1 - Use of Random Forest for portfolio construction:
Has the following components:
1) Obtaining all relevant data (using S&P100 as the 'universe')
2) Training and testing a Machine learning model to predict future changes of stock price
3) USe the model to construct and re-balance stock portfolio. Evaluation of the portfolio performance.


### Plot 2 - USer interaction for portfolio construction:
1) Uses the data from Plot 1
2) Implements reversal strategy, but takes specific inputs from the user (e.g. top returns(t-1) quantile,
threshold for revenue growth, and threshold for EBIT margin. 
3) Constructs and re-balances stock portfolio, based on the considered rules. Runs evaluation of the portfolio performance.

### my_project.py - Complete code for generation of the results
