## Workshop - Regression-Based Classification

Does `statsmodels` marginal effect use the average of covariates or the average predicted values? 
- Use the class data.
- Show your work.

Load the necessary packages and data:


```python
Karan Indian food last time January
Yu Hot pot two weeks ago
Yuman Lamb rack last weekend
```


```python
import numpy as np
import pandas as pd
from tqdm import tqdm # progress bar

import statsmodels.api as sm
from sklearn import linear_model as lm
import statsmodels.formula

from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc = {'axes.titlesize': 24,
             'axes.labelsize': 20,
             'xtick.labelsize': 12,
             'ytick.labelsize': 12,
             'figure.figsize': (8, 4.5)})
sns.set_style("white") # for plot_confusion_matrix()

df = pd.read_pickle('/Users/yumanwu/OneDrive/大学/2021 Spring/ECON 490/Lecture Notes/OLS & Regularization/class_data(4).pkl')
```

Fit a logistic regression using either `sm.Logit()` or `smf.logit()`.


```python
df_prepped = df.drop(columns = ['urate_bin', 'year']).join([
    pd.get_dummies(df['urate_bin'], drop_first = True),
    pd.get_dummies(df.year, drop_first = True)    
])
y = df_prepped['pos_net_jobs'].astype(float)
x = df_prepped.drop(columns = 'pos_net_jobs')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 2/3, random_state = 490)

x_train_std = x_train.apply(lambda x: (x - np.mean(x))/np.std(x), axis = 0)
x_test_std  = x_test.apply(lambda x: (x - np.mean(x))/np.std(x), axis = 0)

x_train_std = sm.add_constant(x_train_std)
x_test_std  = sm.add_constant(x_test_std)
x_train     = sm.add_constant(x_train)
x_test      = sm.add_constant(x_test)
```


```python
fit_logit = sm.Logit(y_train, x_train).fit()
```

    Optimization terminated successfully.
             Current function value: 0.599666
             Iterations 6


Get the marginal effects (`.get_margeff()`). Print the summary (`.summary()`).


```python
fit_logit.summary2()
fit_logit.get_margeff().summary()
```




<table class="simpletable">
<caption>Logit Marginal Effects</caption>
<tr>
  <th>Dep. Variable:</th> <td>pos_net_jobs</td>
</tr>
<tr>
  <th>Method:</th>            <td>dydx</td>    
</tr>
<tr>
  <th>At:</th>               <td>overall</td>  
</tr>
</table>
<table class="simpletable">
<tr>
          <th></th>             <th>dy/dx</th>    <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>pct_d_rgdp</th>        <td>    0.0033</td> <td>    0.000</td> <td>   11.150</td> <td> 0.000</td> <td>    0.003</td> <td>    0.004</td>
</tr>
<tr>
  <th>emp_estabs</th>        <td>    0.0079</td> <td>    0.001</td> <td>   13.394</td> <td> 0.000</td> <td>    0.007</td> <td>    0.009</td>
</tr>
<tr>
  <th>estabs_entry_rate</th> <td>    0.0401</td> <td>    0.001</td> <td>   38.375</td> <td> 0.000</td> <td>    0.038</td> <td>    0.042</td>
</tr>
<tr>
  <th>estabs_exit_rate</th>  <td>   -0.0336</td> <td>    0.001</td> <td>  -28.236</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.031</td>
</tr>
<tr>
  <th>pop</th>               <td> 3.029e-08</td> <td> 1.09e-08</td> <td>    2.782</td> <td> 0.005</td> <td> 8.95e-09</td> <td> 5.16e-08</td>
</tr>
<tr>
  <th>pop_pct_black</th>     <td>   -0.0008</td> <td>    0.000</td> <td>   -3.957</td> <td> 0.000</td> <td>   -0.001</td> <td>   -0.000</td>
</tr>
<tr>
  <th>pop_pct_hisp</th>      <td>    0.0013</td> <td>    0.000</td> <td>    6.678</td> <td> 0.000</td> <td>    0.001</td> <td>    0.002</td>
</tr>
<tr>
  <th>lfpr</th>              <td>    0.0003</td> <td>    0.000</td> <td>    1.037</td> <td> 0.300</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>density</th>           <td>  1.59e-06</td> <td> 1.77e-06</td> <td>    0.898</td> <td> 0.369</td> <td>-1.88e-06</td> <td> 5.06e-06</td>
</tr>
<tr>
  <th>lower</th>             <td>    0.0709</td> <td>    0.007</td> <td>   10.278</td> <td> 0.000</td> <td>    0.057</td> <td>    0.084</td>
</tr>
<tr>
  <th>similar</th>           <td>    0.0353</td> <td>    0.007</td> <td>    4.788</td> <td> 0.000</td> <td>    0.021</td> <td>    0.050</td>
</tr>
<tr>
  <th>2003</th>              <td>    0.2232</td> <td>    0.014</td> <td>   15.973</td> <td> 0.000</td> <td>    0.196</td> <td>    0.251</td>
</tr>
<tr>
  <th>2004</th>              <td>    0.2627</td> <td>    0.014</td> <td>   18.459</td> <td> 0.000</td> <td>    0.235</td> <td>    0.291</td>
</tr>
<tr>
  <th>2005</th>              <td>    0.2643</td> <td>    0.015</td> <td>   18.120</td> <td> 0.000</td> <td>    0.236</td> <td>    0.293</td>
</tr>
<tr>
  <th>2006</th>              <td>    0.3600</td> <td>    0.015</td> <td>   23.839</td> <td> 0.000</td> <td>    0.330</td> <td>    0.390</td>
</tr>
<tr>
  <th>2007</th>              <td>    0.1184</td> <td>    0.014</td> <td>    8.467</td> <td> 0.000</td> <td>    0.091</td> <td>    0.146</td>
</tr>
<tr>
  <th>2008</th>              <td>    0.1862</td> <td>    0.014</td> <td>   13.189</td> <td> 0.000</td> <td>    0.159</td> <td>    0.214</td>
</tr>
<tr>
  <th>2009</th>              <td>   -0.1147</td> <td>    0.017</td> <td>   -6.834</td> <td> 0.000</td> <td>   -0.148</td> <td>   -0.082</td>
</tr>
<tr>
  <th>2010</th>              <td>    0.0389</td> <td>    0.015</td> <td>    2.644</td> <td> 0.008</td> <td>    0.010</td> <td>    0.068</td>
</tr>
<tr>
  <th>2011</th>              <td>    0.2585</td> <td>    0.014</td> <td>   18.345</td> <td> 0.000</td> <td>    0.231</td> <td>    0.286</td>
</tr>
<tr>
  <th>2012</th>              <td>    0.3308</td> <td>    0.014</td> <td>   22.889</td> <td> 0.000</td> <td>    0.302</td> <td>    0.359</td>
</tr>
<tr>
  <th>2013</th>              <td>    0.2510</td> <td>    0.014</td> <td>   17.594</td> <td> 0.000</td> <td>    0.223</td> <td>    0.279</td>
</tr>
<tr>
  <th>2014</th>              <td>    0.3165</td> <td>    0.014</td> <td>   21.919</td> <td> 0.000</td> <td>    0.288</td> <td>    0.345</td>
</tr>
<tr>
  <th>2015</th>              <td>    0.3389</td> <td>    0.015</td> <td>   22.840</td> <td> 0.000</td> <td>    0.310</td> <td>    0.368</td>
</tr>
<tr>
  <th>2016</th>              <td>    0.2659</td> <td>    0.014</td> <td>   18.345</td> <td> 0.000</td> <td>    0.238</td> <td>    0.294</td>
</tr>
<tr>
  <th>2017</th>              <td>    0.2295</td> <td>    0.014</td> <td>   16.001</td> <td> 0.000</td> <td>    0.201</td> <td>    0.258</td>
</tr>
<tr>
  <th>2018</th>              <td>    0.3054</td> <td>    0.015</td> <td>   20.812</td> <td> 0.000</td> <td>    0.277</td> <td>    0.334</td>
</tr>
</table>



***
# Covariate Averages
$$
\frac{\partial p(x_i)}{\partial \beta_1} \approx \frac{e^{\hat{\beta}_0 + \bar{x}\hat{\beta}_1 + \bar{x}\hat{\beta_2}}}{(1 + e^{\hat{\beta}_0 + \bar{x}\hat{\beta}_1 + \bar{x}\hat{\beta_2}})^2}\hat{\beta}
$$


```python
beta = fit_logit.params
avgs = np.array([1.,np.mean(df.pct_d_rgdp),np.mean(df.estabs_entry_rate)])
```


```python

```

***
# Predicted values Averages
$$
\frac{\partial p(x_i)}{\partial \beta_1} \approx \frac{1}{n} \sum_{i=1}
^n \frac{e^{\hat{y}_i}}{1 + e^{\hat{y}_i}}\hat{\beta}
$$


```python

```


```python

```

*** 
# Interpretaton

Interpret the marginal effect on one feature.


