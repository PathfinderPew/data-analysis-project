import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── 1. Load data ──────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# ── 2. Define features ────────────────────────────────────
features = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
            'Electrical_Conductivity', 'Temperature_C', 'Humidity',
            'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh',
            'Field_Area_hectare', 'Previous_Irrigation_mm']

X_train = train[features]
y_train = train['Irrigation_Need']
X_test = test[features]

# ── 3. Train model ────────────────────────────────────────
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Done!")

# ── 4. Make predictions ───────────────────────────────────
predictions = model.predict(X_test)

# ── 5. Save submission ────────────────────────────────────
submission = pd.DataFrame({
    'id': test['id'],
    'Irrigation_Need': predictions
})
submission.to_csv('data/submission.csv', index=False)
print("Submission saved to data/submission.csv")
print(submission['Irrigation_Need'].value_counts())