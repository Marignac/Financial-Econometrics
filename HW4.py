import numpy as np
import pandas as pd
from datetime import datetime
import math
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

os.chdir(r'D:\Dokumente\Studium\Master\Université de Genève\Kurse\Financial Econometrics\Homeworks\HW4')
SuPI = pd.read_excel("TP4.xls", 'S&P100Index', skiprows = 4, nrows=524,  usecols = 'A:C')
SuPC = pd.read_excel("TP4.xls", 'S&P100Constituents', skiprows = 3, nrows=524,  usecols = 'A:CR')
TB3M = pd.read_excel("TP4.xls", 'TBill3Months', skiprows = 4, nrows=524,  usecols = 'A:B')
FaFr = pd.read_excel("TP4.xls", 'FamaFrenchPortfolios', skiprows = 21, nrows=524,  usecols = 'A:G')

# 1.
#np.cov()
#pd.DataFrame(list(range(1,523)), SuPI.iloc[:,1], SuPC.iloc[:,3])
#pd.DataFrame(list(SuPI.iloc[:,1]), list(SuPC.iloc[:,3]))
#np.array(SuPI.iloc[:,1], SuPC.iloc[:,3])
#np.cov(list(SuPI.iloc[:,1]), list(SuPC.iloc[:,3]))

SuPC = SuPC.tail(-1)
returns = SuPC.copy(deep = True)
returns = returns.assign(SuP500 = SuPI.iloc[:,2])
returns.drop(columns=returns.columns[0], axis=1,  inplace=True)
#mr = (SuPI.iloc[:,2].shift(1) - SuPI.iloc[:,2]) / SuPI.iloc[:,2]

for i in range(0,len(returns.columns)-1):
  returns.iloc[:,i] = (SuPC.iloc[:,i+1].shift(1) - SuPC.iloc[:,i+1]) / SuPC.iloc[:,i+1]
returns.iloc[:,-1] = list((SuPI.iloc[:,2].shift(1) - SuPI.iloc[:,2]) / SuPI.iloc[:,2])
#returns = returns.tail(-1)

np.cov(list(SuPI.iloc[:,2]), list(SuPC.iloc[1:,3]))[0,1]

def beta(market, stock):
  cov = np.cov(market, stock)[0,1]
  var = np.var(market)
  b = cov/var
  return b

parametersData = {"Name": SuPC.columns[1:len(SuPC.columns)],
                  "Beta": np.zeros(len(SuPC.columns)-1),
                  "Alpha": np.zeros(len(SuPC.columns)-1),
                  "TestStatistic": np.zeros(len(SuPC.columns)-1),
                  "Reject": np.zeros(len(SuPC.columns)-1)}
parameters = pd.DataFrame(parametersData, index = list(range(1,len(SuPC.columns))))

maxDate = 0
for i in range(1,len(SuPC)):
  if SuPC.iloc[i,0] > datetime.strptime("08-15-2001", '%m-%d-%Y').date():
    maxDate = i + 1
    break
    
1
for i in range(1,len(returns.columns)):
  parameters.iloc[i-1,1] = beta(list(returns.iloc[1:maxDate,-1]), list(returns.iloc[1:maxDate,i-1]))
parameters

# 2.
# a = E[Yi] - beta * E[Market]
def alpha(market, stock, beta):
  EY = np.mean(stock)
  EM = np.mean(market)
  a = EY - beta * EM
  return a

for i in range(1,len(returns.columns)):
  parameters.iloc[i-1,2] = alpha(returns.iloc[maxDate:,-1], returns.iloc[maxDate:,i-1], parameters.iloc[i-1,1])
  parameters.iloc[i-1,3] = math.sqrt(len(returns.iloc[maxDate:,i-1])) * parameters.iloc[i-1,2] / np.std(returns.iloc[maxDate:,i])
  parameters.iloc[i-1,4] = parameters.iloc[i-1,3] > t.ppf(0.975, len(returns.iloc[maxDate:,i-1]))
  
parameters

# 2.
model = LinearRegression().fit(x, y)
model.intercept

estimates = {"Period": list(range(0,len(returns))),
              "Psi1": np.zeros(len(returns)),
              "Psi2": np.zeros(len(returns))}
                  
estimates = pd.DataFrame(estimates, index = list(range(0,len(returns))))


for i in range(1,len(returns)):
  model = LinearRegression().fit(np.array(returns.iloc[i,:len(returns.columns)-1]).reshape((-1, 1)), np.array(parameters.iloc[:,1]))
  estimates.iloc[i,1] = model.intercept_
  estimates.iloc[i,2] = model.coef_

np.mean(estimates.iloc[i,1])
np.mean(estimates.iloc[i,2])

# 12/11/1992–14/11/1996
# 1.
maxDate = 0
for i in range(0,len(SuPC)):
  if SuPC.iloc[i,0] > datetime.strptime("11-14-1996", '%m-%d-%Y').date():
    maxDate = i - 1
    break

parametersData = {"Name": SuPC.columns[1:len(SuPC.columns)],
                  "Beta": np.zeros(len(SuPC.columns)-1),
                  "1996-2000": np.zeros(len(SuPC.columns)-1),
                  "AverageReturn": 0}
parameters = pd.DataFrame(parametersData, index = list(range(1,len(SuPC.columns))))

for i in range(1,len(returns.columns)):
  parameters.iloc[i-1,1] = beta(list(returns.iloc[1:maxDate,-1]), list(returns.iloc[1:maxDate,i-1]))
parameters

# 2.
parameters_ordered = parameters.sort_values(by = "Beta")

parametersSplit = np.array_split(parameters_ordered, 10)
parametersSplit[0]

# 3.
minDate = 0
maxDate = 0
for i in range(0,len(SuPC)):
  if SuPC.iloc[i,0] == datetime.strptime("11-21-1996", '%m-%d-%Y').date():
    minDate = i
  if SuPC.iloc[i,0] == datetime.strptime("11-16-2000", '%m-%d-%Y').date():
    maxDate = i
  
for i in range(1,len(returns.columns)):
  parameters.iloc[i-1,2] = beta(list(returns.iloc[minDate:maxDate,-1]), list(returns.iloc[minDate:maxDate,i-1]))
  parameters.iloc[i-1,3] = np.mean(returns.iloc[minDate:maxDate,i])
parameters

parameters_ordered = parameters.sort_values(by = "1996-2000")

parametersSplit = np.array_split(parameters_ordered, 10)

portfolios = {"Portfolio": list(range(1,11)),
              "Beta": 0,
              "ExcessReturns": 0}
portfolios = pd.DataFrame(portfolios, index = list(range(1,11)))

for i in range(0,10):
  portfolios.iloc[i,1] = np.mean(parametersSplit[i].iloc[:,2])
  portfolios.iloc[i,2] = np.mean(parametersSplit[i].iloc[:,3])
portfolios

# 4.
exactDate = 0
for i in range(0,len(SuPC)):
  if SuPC.iloc[i,0] == datetime.strptime("11-23-2000", '%m-%d-%Y').date():
    exactDate = i
    break

model = LinearRegression().fit(np.array(returns.iloc[exactDate,:len(returns.columns)-1]).reshape((-1, 1)), np.array(parameters.iloc[:,1]))
model.intercept_
model.coef_

minDate = 0
maxDate = 0
for i in range(0,len(SuPC)):
  if SuPC.iloc[i,0] == datetime.strptime("11-23-2000", '%m-%d-%Y').date():
    minDate = i
  if SuPC.iloc[i,0] == datetime.strptime("11-14-2002", '%m-%d-%Y').date():
    maxDate = i

estimates = {"Period": list(range(0, maxDate-minDate)),
              "Psi1": 0,
              "Psi2": 0}
                  
estimates = pd.DataFrame(estimates, index = list(range(maxDate-minDate)))
estimates

pfReturns = {"Period": list(range(0,maxDate-minDate)),
              "PF1" : 0,
              "PF2" : 0,
              "PF3" : 0,
              "PF4" : 0,
              "PF5" : 0,
              "PF6" : 0,
              "PF7" : 0,
              "PF8" : 0,
              "PF9" : 0,
              "PF10" : 0}
              
pfReturns = pd.DataFrame(pfReturns, index = list(range(maxDate-minDate)))             
ret = 0
number = 0
for i in range(minDate,maxDate):
  for j in range(len(portfolios)):
    for k in parametersSplit[j].index:
      #print("i:", i, "     j:", j, ",     k:", k)
      ret = ret + returns.iloc[i,k]
      number = number + 1
    pfReturns.iloc[i-minDate,j+1] = ret/number
    ret = 0
    number = 0
pfReturns

for i in range(0,len(estimates)):
  model = LinearRegression().fit(np.array(pfReturns.iloc[i,1:]).reshape((-1, 1)), np.array(portfolios.iloc[:,1]))
  estimates.iloc[i,1] = model.intercept_
  estimates.iloc[i,2] = model.coef_

# Testing alpha
(np.mean(estimates.iloc[:,1]) / np.std(estimates.iloc[:,1]))*math.sqrt(maxDate-minDate)
t.ppf(0.975,maxDate-minDate-1)
(np.mean(estimates.iloc[:,2]) / np.std(estimates.iloc[:,2]))*math.sqrt(maxDate-minDate)
t.ppf(0.95,maxDate-minDate-1)
