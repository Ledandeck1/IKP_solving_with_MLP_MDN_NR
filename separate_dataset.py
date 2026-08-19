import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Splitting Dataset...")

# 1. Load Data[cite: 4]
df = pd.read_csv(os.path.join('dataset_raw.csv'))

# Create a list of the 21 coordinate column names dynamically
#coord_cols = [f'{axis}{i}' for i in range(7) for axis in ['x', 'y', 'z']]
coord_cols = ['x', 'y', 'z']

# X = Features (Coordinates), Y = Labels (Hardware Angles)[cite: 4]
X = df[coord_cols].values
y = df[['theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5']].values

# 2. Split 80-10-10[cite: 4]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print(f"Inputs (Features): {X_train.shape[1]}")
print(f"Outputs (Targets): {y_train.shape[1]}")

# 3. Scaling (StandardScaler)[cite: 4]
scaler_X = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)

# 4. Save Processed Data[cite: 4]
np.savez(os.path.join('dataset_processed.npz'),
         X_train=X_train_s, y_train=y_train,
         X_val=X_val_s, y_val=y_val,
         X_test=X_test_s, y_test=y_test)

# Save scalers for inference[cite: 4]
joblib.dump(scaler_X, os.path.join('scaler_X.pkl'))

print("Data split and scaled successfully.")