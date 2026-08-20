import numpy as np
import pandas as pd
import os
import csv
from forwardkinematics import ForwardKinematic

# ==========================================
# CONFIGURATION
# ==========================================



# ==========================================
# DATASET GENERATION
# ==========================================
print("Generating Dataset with Geometric FK...")
output_file = os.path.join('dataset_raw.csv')

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['theta0', 'theta1', 'theta2', 'theta3', 'x', 'y', 'z'])

    for _ in range(350000):
        # 1. Generate Random Hardware Angles (Absolute Physical Angles)

        # t1 = np.random.uniform(15.0, 165.0)
        # t2 = np.random.uniform(15.0, 115.0)
        # t3 = np.random.uniform(0.0, 90.0)
        # t4 = np.random.uniform(0.0, 90.0)

        t1 = np.random.uniform(15.0, 165.0)
        t2 = np.random.uniform(15.0, 115.0)
        t3 = np.random.uniform(-90.0, 0.0)
        t4 = np.random.uniform(-90.0, 0.0)

        th0 = np.deg2rad(t1)  # Base
        th1 = np.deg2rad(t2)  # Shoulder
        th2 = np.deg2rad(t3)  # Elbow (0=Vertical, 90=Horizontal)
        th3 = np.deg2rad(t4)  # Wrist (0=Vertical, 90=Horizontal, 150=Backward)

        #th0_deg = np.random.uniform(15.0, 165.0)  # Base
        #th1_deg = np.random.uniform(15.0, 115.0)  # Shoulder
        #th2_deg = np.random.uniform(0.0, 90.0)  # Elbow (0=Vertical, 90=Horizontal)
        #th3_deg = np.random.uniform(0.0, 150.0)  # Wrist (0=Vertical, 90=Horizontal, 150=Backward)

        # 2. Calculate Position
        xyz = ForwardKinematic(th0, th1, th2, th3)

        # 3. Save Hardware Angles + XYZ
        writer.writerow([th0, th1, th2, th3] + list(xyz))

print(f"Dataset saved to {output_file}")

df = pd.read_csv(os.path.join('dataset_raw.csv'))

# X = Features (Coordinates), Y = Labels (Hardware Angles)
X = df[['x', 'y', 'z']].values
y = df[['theta0', 'theta1', 'theta2', 'theta3']].values

# 2. Split 80-10-10
# First split: 80% Train, 20% Temp
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: Split 20% Temp into 10% Val, 10% Test
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 3. Scaling (StandardScaler)
# Fit on TRAIN only
scaler_X = StandardScaler()
#scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
#y_train_s = scaler_y.fit_transform(y_train)

X_val_s = scaler_X.transform(X_val)
#y_val_s = scaler_y.transform(y_val)

X_test_s = scaler_X.transform(X_test)
#y_test_s = scaler_y.transform(y_test)

# 4. Save Processed Data
# np.savez(os.path.join('dataset_processed.npz'),
#          X_train=X_train_s, y_train=y_train_s,
#          X_val=X_val_s, y_val=y_val_s,
#          X_test=X_test_s, y_test=y_test_s)

np.savez(os.path.join('dataset_processed.npz'),
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)


# Save scalers for inference
joblib.dump(scaler_X, os.path.join('scaler_X.pkl'))
#joblib.dump(scaler_y, os.path.join('scaler_y.pkl'))

print("Data split and scaled successfully.")