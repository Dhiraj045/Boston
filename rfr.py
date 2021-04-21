import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns = ["MEDV"])

rfr = RandomForestRegressor()
rfr.fit(X,Y)

pickle.dump(rfr, open('rfr_algo.pkl', 'wb'))
