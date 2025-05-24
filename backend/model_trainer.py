import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('../data/Training.csv')

# Drop the label column from features
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Ensure X is numeric
X = X.apply(pd.to_numeric)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, '../backend/disease_model.pkl')
joblib.dump(le, '../backend/label_encoder.pkl')

print("✅ Model training complete. Files saved.")
