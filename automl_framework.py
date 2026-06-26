import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

print("AutoML Framework Running...\n")

# Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

results = {}

print("\nTraining Models...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy

    print(f"{name} Accuracy : {accuracy*100:.2f}%")

# Best Model
best_model = max(results, key=results.get)

print("\nBest Model :", best_model)
print("Best Accuracy :", round(results[best_model]*100,2),"%")

# Plot
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8,5))

plt.bar(results.keys(), results.values())

plt.ylabel("Accuracy")

plt.title("AutoML Model Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("outputs/model_comparison.png")

plt.show()

print("\nProject Completed Successfully!")