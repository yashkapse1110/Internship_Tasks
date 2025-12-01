import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("Iris.csv") 

if "Id" in df.columns:
    df = df.drop(columns=["Id"])
print("Dataset Loaded Successfully")
print(df.head())

X = df.drop("Species", axis=1).values    
y = df["Species"].values                

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
y_pred_logreg = logreg.predict(X_test_scaled)

print("\nModel Evaluation:\n")

print("KNN Accuracy:", (round(accuracy_score(y_test, y_pred_knn), 2 )*100))
print("Logistic Regression Accuracy:",( round(accuracy_score(y_test, y_pred_logreg), 2))*100)

print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_logreg))


