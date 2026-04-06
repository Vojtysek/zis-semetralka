import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset = arff.loadarff('speeddating.arff')
df = pd.DataFrame(dataset[0]).apply(lambda x: x.str.decode('utf-8') if x.dtype == "object" else x)

df = df.drop(columns=['expected_num_interested_in_me', 'expected_num_matches'])

cols_to_fix = ['age', 'age_o', 'sports', 'reading', 'yoga'] 
for col in cols_to_fix:
    df[col] = df[col].fillna(df[col].median())

df['funny_o'] = df['funny_o'].fillna(df['funny_o'].mode()[0])

X = df.drop(columns=['match', 'decision', 'decision_o'])
y = df['match'].astype(int)

X = pd.get_dummies(X, drop_first=True)

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on the test set
y_pred = model.predict(X_test)

# Show the 'True Positives' vs 'False Positives'
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')