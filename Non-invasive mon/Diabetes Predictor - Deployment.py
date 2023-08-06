import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#ins%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

# Loading the dataset
df = pd.read_csv("yayayay-2.csv")

categorical_columns = ['Gender', 'Pregnancies', 'Smoking', 'Alcohol Consumption','Blood Pressure (mmHg)']

# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split the dataset into features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)
print(y_train.shape)

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)
classifier_pred = classifier.predict(X_test_scaled)
classifier_accuracy = accuracy_score(y_test, classifier_pred)
classifier_conf_matrix = confusion_matrix(y_test, classifier_pred)
tree_classification_report = classification_report(y_test, classifier_pred)

print("Random Forest :")
print(f"Accuracy: {classifier_accuracy:.2f}")
print("Confusion Matrix:")
print(classifier_conf_matrix)

#filename = 'diabetes-prediction-rfc-model.pkl'
#with open(filename, 'wb') as file:
 #   classifier = pickle.load(file)

filename = 'ALL.pkl'
pickle.dump(classifier, open(filename, 'wb'))