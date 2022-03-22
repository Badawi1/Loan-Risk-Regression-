import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

def onehot_encode(df, column_with_perfix):
    df = df.copy()
    for column, prefix in column_with_perfix:
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df =df.drop(column, axis=1)
    return df

def preprocessing_inputs(df):
    df = df.copy()
    df = df.drop("CreditGrade", axis=1)
    df = df.drop("TotalProsperPaymentsBilled", axis=1)
    df = df.drop("ListingNumber", axis=1)
    df = df.drop("BorrowerState", axis=1)
    df = df.dropna(subset=['LoanRiskScore'])
    df['DebtToIncomeRatio'].fillna(value=df['DebtToIncomeRatio'].mean(), inplace=True)
    df['EmploymentStatusDuration'].fillna(value=df['EmploymentStatusDuration'].mean(), inplace=True)

    df["IncomeRange"].replace(
        {"$1-24,999": 12500, "$25,000-49,999": 37500, "$50,000-74,999": 62500, "$75,000-99,999": 87500,
         "$100,000+": 100000, "Not employed": 0, '$0 ': 0}, inplace=True)

    df["IsBorrowerHomeowner"] = df["IsBorrowerHomeowner"].astype(int)


    df = onehot_encode(
        df,
        column_with_perfix=[
            ('EmploymentStatus', 'E'),
            ('LoanStatus', 'L')
        ]
    )
    return df


data = pd.read_csv("LoanRiskScore.csv")
df = preprocessing_inputs(data)
Y=df["LoanRiskScore"]
X= df.drop("LoanRiskScore", axis=1)

#Feature Selection
sel = SelectKBest(mutual_info_regression, k=15)
sel.fit(X,Y)
new_X = sel.transform(X)

#saving features
#with open('selectbest_15_reg','wb') as f:
    #pickle.dump(sel,f)

X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size = 0.20,random_state=0)

forest=RandomForestRegressor()
scores = cross_val_score(forest,X_train, y_train, scoring='neg_mean_squared_error', cv=5)
model_1_score = abs(scores.mean())
print("Random Forest Model Cross Validation Score Is "+ str(model_1_score))
forest.fit(X_train,y_train)

#saving model
#with open('rf_reg','wb') as f:
    #pickle.dump(forest,f)

prediction = forest.predict(X_test)
print("MSE ",metrics.mean_squared_error(y_test,prediction))
print("RMSE ",sqrt(metrics.mean_squared_error(y_test,prediction)))
SS_Residual = sum((y_test-prediction)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)
print("R2 Score ", r_squared)
print("Adjusted R2 Score ", adjusted_r_squared)