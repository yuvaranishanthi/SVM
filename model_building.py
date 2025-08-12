import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
csv_file = "Car_Evaluation Original Dataset.csv"  # Must be in same folder
df = pd.read_csv(csv_file)

# Use only important features based on actual column names
important_features = ['Buying_Price', 'Maintenance_Price', 'safety']
target_col = 'class'

X = df[important_features].copy()
y = df[target_col].copy()

# Encode categorical features
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/svm_model.pkl")
joblib.dump(important_features, "model/features.pkl")
joblib.dump(label_encoders, "model/encoders.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")

# Save dropdown options for form
dropdown_options = {}
for col in df[important_features]:
    dropdown_options[col] = sorted(df[col].unique().tolist())
joblib.dump(dropdown_options, "model/dropdown_options.pkl")

print("✅ SVM model trained successfully using only important features.")
