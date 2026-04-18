import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
df = pd.read_csv('updated_parking_data.csv')

# 2. Data Preprocessing (Encoding Categorical Data)
# Converting slot IDs into numerical values for the model
le = LabelEncoder()
df['slot_id_encoded'] = le.fit_transform(df['slot_id'])

# 3. Define Features (X) and Target (y)
# Features used: Slot ID, Day of the Week, and Hour
X = df[['slot_id_encoded', 'day_of_week (1 for Monday, 7 for Sunday)', 'hour_of_day']]
y = df['field2 (Avaliabality)']

# 4. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model (Random Forest Classifier)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Prediction and Evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 7. Print Results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
