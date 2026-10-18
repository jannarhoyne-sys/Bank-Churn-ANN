import pandas as pd
import tensorflow as tf
print(tf.__version__)

dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset.shape)
dataset.head()

# Check for missing values
print(dataset.isnull().sum().sum(), "missing values\n")

# Churn rate for each category (0.20 = 20% of that group left)
for col in ['Geography', 'Gender', 'IsActiveMember', 'NumOfProducts', 'HasCrCard']:
    print(dataset.groupby(col)['Exited'].mean().round(3), '\n')

# Correlation of numeric features with Exited
numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited']
print(dataset[numeric].corr()['Exited'].sort_values())

X = dataset.iloc[:, 3:-1].values   # CreditScore through EstimatedSalary (10 features)
y = dataset.iloc[:, -1].values     # Exited
print(X[0])


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X[0])

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X[0])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=29, stratify=y
)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


tf.keras.utils.set_random_seed(29)

ann = tf.keras.models.Sequential([
    tf.keras.Input(shape=(12,)),                          # 12 features in
    tf.keras.layers.Dense(units=6, activation='relu'),    # hidden layer 1
    tf.keras.layers.Dense(units=6, activation='relu'),    # hidden layer 2
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # output layer
])

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.summary()

ann.fit(X_train, y_train, batch_size=32, epochs=100)

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

# Average profile: customers who stayed (0) vs. left (1)
profile = dataset.groupby('Exited')[['CreditScore', 'Age', 'Tenure', 'Balance',
                                     'NumOfProducts', 'IsActiveMember', 'EstimatedSalary']].mean()
profile.round(2)

cols = profile.columns
effect = (profile.loc[1] - profile.loc[0]) / dataset[cols].std()
effect.round(2).sort_values()



