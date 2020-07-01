import pandas as pd
import pickle

# load data
df = pd.read_csv('winequality-red.csv', sep=';', 
                 usecols=["fixed acidity", 'volatile acidity','chlorides',
                          'free sulfur dioxide', 'pH', 'alcohol', 'quality'])

# define dependent var and independent vars
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# split data as train and tes data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# create regressor for prediction
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# train model
regressor.fit(X_train, y_train)

# dump regressor as pickle file
pickle.dump(regressor, open('model.pkl','wb'))

# test load model file
# model = pickle.load(open('model.pkl','rb'))

# test prediction
# print(model.predict([[ 7.4, 0.7,0.076,11,3.51,9.4]]))