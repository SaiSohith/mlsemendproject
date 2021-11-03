import pandas as pd
import numpy as np
life_expectancy_df = pd.read_csv('Life_Expectancy_Data_1.csv')


life_expectancy_df.isnull().sum()[np.where(life_expectancy_df.isnull().sum() != 0)[0]]

life_expectancy_df = life_expectancy_df.apply(lambda x: x.fillna(x.mean()),axis=0)

life_expectancy_df.isnull().sum()[np.where(life_expectancy_df.isnull().sum() != 0)[0]]


X = life_expectancy_df.drop(columns = ['Life expectancy '])
y = life_expectancy_df[['Life expectancy ']]

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept = True)
model.fit(X_train, y_train)

# y_predict = model.predict(X_test)

y_test_orig = scaler_y.inverse_transform(y_test)

from sklearn.tree import DecisionTreeRegressor 
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

from sklearn.ensemble import  RandomForestRegressor
random_forest_model = RandomForestRegressor()
random_forest_fit = random_forest_model.fit(X_train, y_train)


def mod(xtest):
    xtest = scaler_X.transform(xtest)
    ypredict = model.predict(xtest)
    
    y_predict_orig = scaler_y.inverse_transform(ypredict)
    return y_predict_orig.tolist()
