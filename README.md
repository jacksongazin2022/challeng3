```python
import pandas as pd
import numpy as np
import sklearn
import scipy
import matplotlib
import statsmodels
```

## Question 2:  Load the SBA data into Python, using the pandas.read_csv functio


```python
df = pd.read_csv("https://sta712-f23.github.io/homework/sba_small.csv")
```

## Question 3: List the variables in the SBA data that you will use to answer the research question above.

The variables I will be using to answer my question are MIS_Status, GrAppv, NewExist, and UrbanRural.


```python
df.columns
unique_values = df['MIS_Status'].unique()
unique_values
```




    array(['P I F', 'CHGOFF', nan], dtype=object)



## Question 4: Using the MIS_Status column, create a new column in your SBA data called Default, which is equal to 1 if the loan was charged off (i.e., the borrower defaulted), and 0 if the loan was paid in full (the borrower did not default).


```python
df['Default'] = df['MIS_Status'].apply(lambda x: 1 if x == 'CHGOFF' else 0)
unique_values = df['Default'].unique()
unique_values
```




    array([0, 1])



## Question 5: Create a new column in your SBA data called Amount which is the loan amoun


```python
df['Amount'] = df['GrAppv']
```

## Question 6: In R, categorical variables automatically get converted to indicator variables when we fit a logistic regression model. This is not true in Python; part of our data pre-processing is to create the indicator variables we need. This can be done with the sklearn.preprocessing.OneHotEncoder class. Create a new dataset called sba_encoded which contains your Amount column from Question 5, and one-hot encodings of UrbanRural and NewExist. (For the purposes of this activity, we will ignore any potential interactions between the explanatory variables). Hint: you will probably want to use drop = ‘first’ in your one-hot encoding!


```python
from sklearn.preprocessing import OneHotEncoder
sba_encoded = df[['Amount']]
encoder = OneHotEncoder(drop='first', sparse_output = False)
encoded_cols = encoder.fit_transform(df[['UrbanRural', 'NewExist']])
encoded_column_names = encoder.get_feature_names_out(['UrbanRural', 'NewExist'])
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_column_names)
sba_encoded = pd.concat([sba_encoded, encoded_df], axis=1)
sba_encoded
encoded_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UrbanRural_1</th>
      <th>UrbanRural_2</th>
      <th>NewExist_1</th>
      <th>NewExist_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 4 columns</p>
</div>



## Question 7: Using the sklearn.linear_model.LogisticRegression class, the Default column from Question 4, and the sba_encoded data from Question 6, fit a logistic regression model and report the estimated coefficients. Hint: you will want to use penalty = ‘none’ when creating the model.


```python
from sklearn.linear_model import LogisticRegression
X = sba_encoded
Y = df['Default']
model = LogisticRegression(penalty=None)
model.fit(X, Y)
coefficients = model.coef_

model.intercept_
```




    array([-11.2010176])




```python
coefficients
```




    array([[-0.28297796,  1.34052561,  1.0436726 , 11.86946659, 11.97611109]])



|          | Estimate      |
|----------|---------------|
| intercept | -11.2010176  |
| Amount | -0.28297796  |
| UrbanRural1| 1.34052561   |
| UrbanRural2| 1.0436726    |
| NewExist1  | 11.86946659  |
|NewExist2  | 11.97611109  |

We have that $\hat{\beta}_0 = -11.2010176] \hat{\beta}_1 =-0.28297796, \hat{\beta}_2 =   1.34052561, \hat{\beta}_3 = 1.0436726$

And that $\hat{\beta}_4= 11.86946659$ and $\hat{\beta_5}= 11.97611109$

## Question 8: Do the estimated coefficients from Question 7 agree with the estimated coefficients for the same model in R? How do your estimated coefficients change when you change the solver in your logistic regression?



Below is my r code and table I got in R. We can see that our estimates are slightly different. 
sba <- read.csv("https://sta712-f23.github.io/homework/sba_small.csv")

str(sba)

# Create a new column 'Default' based on 'MIS_Status'
sba$Default <- ifelse(sba$MIS_Status == 'CHGOFF', 1, 0)

sba$UrbanRural <- as.factor(sba$UrbanRural)
sba$NewExist <- as.factor(sba$NewExist)

model <- glm(Default ~ GrAppv + UrbanRural + NewExist, data = sba, family = "binomial")
options(scipen = 999)
library(xtable)
xtable(model)
|            | Estimate  | Std. Error | z value | Pr($>$$|$z$|$) |
|------------|-----------|------------|---------|----------------|
| (Intercept)| -14.0457  | 188.7731   | -0.07   | 0.9407         |
| GrAppv     | -0.0000   | 0.0000     | -7.16   | 0.0000         |
| UrbanRural1| 1.4524    | 0.1037     | 14.01   | 0.0000         |
| UrbanRural2| 1.1009    | 0.1379     | 7.99    | 0.0000         |
| NewExist1  | 11.7108   | 188.7731   | 0.06    | 0.9505         |
| NewExist2  | 11.8175   | 188.7731   | 0.06    | 0.9501         |

We can see my R estimates are slightly different


```python
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")

X = sba_encoded
Y = df['Default']

solvers = ['newton-cg', 'lbfgs', 'sag', 'saga']

models = {}
for solver in solvers:
    model = LogisticRegression(penalty='none', solver=solver)
    model.fit(X, Y)
    models[solver] = model

# Access and print the coefficients and intercept for each model
for solver, model in models.items():
    coefficients = model.coef_
    intercept = model.intercept_
    print(f"Solver: {solver}")
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

```

    Solver: newton-cg
    Coefficients: [[-0.28303729  1.34050329  1.04379269 12.35163628 12.45830621]]
    Intercept: [-11.68253768]
    Solver: lbfgs
    Coefficients: [[-0.28297796  1.34052561  1.0436726  11.86946659 11.97611109]]
    Intercept: [-11.2010176]
    Solver: sag
    Coefficients: [[-0.28570796  1.33728375  1.04201781  0.55501588  0.66077429]]
    Intercept: [0.14569225]
    Solver: saga
    Coefficients: [[-0.28613045  1.3367916   1.04174977  0.39375691  0.49937578]]
    Intercept: [0.31190541]


We can see heavy variation among the intercept when choosing a different solver. Our GrAppv does not seem to change drastically. However, our NewExist1 and NewExist2 do seem to change a lot with different solvers!

## Question 9: Using the sklearn.metrics.log_loss function, calculate the deviance for your logistic regression model in Python, and compare to the deviance reported by R.


```python
from sklearn.metrics import log_loss


predicted_probabilities = model.predict_proba(X)

deviance = log_loss(Y, predicted_probabilities)
```


```python
deviance
```




    0.4349433354986975




```python
model$deviance
```

Our r deviance is 4362.791 while our python deviance is 0.4349433354986975


## Question 10:  Using your fitted model in Python, perform a hypothesis test to address the first research question above: Is there a relationship between loan amount and the probability the business defaults on the loan, after accounting for whether or not the business is new, and whether it is in an urban or rural environment?


```python
from scipy import stats
amount_coefficient = model.coef_[0][0]
amount_std_error = np.sqrt(np.diag(np.linalg.inv(X.T @ X))[0])


z_score = amount_coefficient / amount_std_error

p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

```

We get a p value of approximately 0 so we reject our hypothesis that there is  a relationship between loan amount and the probability the business defaults on the loan, after accounting for whether or not the business is new, and whether it is in an urban or rural environment.

## Question 11: . As you can see from the previous questions, the scikit-learn module is very good for building and assessing prediction models, but is less useful for doing statistical inference. For example, we don’t get a nice summary table for our model with estimated standard errors, we need to calculate deviance separately, etc. One way to get these nice summaries in Python is with the statsmodels module. Using the statsmodels.GLM class, fit the same logistic regression model as above. Use the .summary() function to report a nice table with the estimated coefficients and standard errors. Hint: make sure to add an intercept column to the sba_encoded data. The statsmodels module does not include an intercept for you.


```python
import statsmodels.api as sm


X_with_intercept = sm.add_constant(sba_encoded)


model = sm.GLM(df['Default'], X_with_intercept, family=sm.families.Binomial())
results = model.fit()


summary = results.summary()


print(summary)
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                Default   No. Observations:                 5000
    Model:                            GLM   Df Residuals:                     4994
    Model Family:                Binomial   Df Model:                            5
    Link Function:                  Logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -2174.7
    Date:                Wed, 08 Nov 2023   Deviance:                       4349.4
    Time:                        16:08:23   Pearson chi2:                 5.01e+03
    No. Iterations:                    21   Pseudo R-squ. (CS):            0.07143
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const          -19.9192   1.75e+04     -0.001      0.999   -3.44e+04    3.44e+04
    Amount          -0.2830      0.030     -9.305      0.000      -0.343      -0.223
    UrbanRural_1     1.3405      0.105     12.767      0.000       1.135       1.546
    UrbanRural_2     1.0438      0.139      7.534      0.000       0.772       1.315
    NewExist_1      20.5883   1.75e+04      0.001      0.999   -3.44e+04    3.44e+04
    NewExist_2      20.6950   1.75e+04      0.001      0.999   -3.44e+04    3.44e+04
    ================================================================================


## Question 12: . Explain why the standard errors for the NewExist and Intercept coefficients are so high. How would we fix that issue?

I might imagine that the standard error is high because of multi-collinearity.  One way to fix this would be to add a regulizer and perform either Ridge Logistic Regression or Lasso Logistic Regression. 

## Question 13:


```python
def quantile_residuals(model, X):
    n = len(X)
    
   
    phat = model.predict(X)
    
    quantile_residuals = np.zeros(n)
    
    for i in range(n):
        if X['NewExist_1'].iloc[i] == 1:
            u = np.random.uniform(phat[i], 1)
            quantile_residuals[i] = norm.ppf(u)
        else:
            u = np.random.uniform(0, 1 - phat[i])
            quantile_residuals[i] = norm.ppf(u)
    
    return quantile_residuals
```


```python
from sklearn.linear_model import LogisticRegression
X = sba_encoded
Y = df['Default']
model = LogisticRegression(penalty=None)
model.fit(X, Y)
coefficients = model.coef_
```


```python
import matplotlib.pyplot as plt

quantile_residuals = quantile_residuals(model, sba_encoded)

# Create a scatterplot of Amount vs. quantile residuals
plt.scatter(sba_encoded['Amount'], quantile_residuals)
plt.xlabel("Amount")
plt.ylabel("Quantile Residuals")
plt.title("Quantile Residual Plot for Amount")
plt.show()

```


    
![png](output_39_0.png)
    



```python

```
