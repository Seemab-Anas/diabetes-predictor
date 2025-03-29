import pandas as pd
import joblib  # Add this to save the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("diabetes.csv")

# 1. Split data into features (X) and labels (Y)
X = df.drop(columns=["Outcome"])
Y = df["Outcome"]

# 2. Train/Test Split (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train SVM Model
model = SVC(kernel="linear")
model.fit(X_train, Y_train)

# 5. Save the model
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler too!

# 6. Make Predictions
Y_pred = model.predict(X_test)

# 7. Evaluate Model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
