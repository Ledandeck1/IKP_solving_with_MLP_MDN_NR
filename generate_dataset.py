import numpy as np
import os
import csv
from forwardkinematics import ForwardKinematic_points
from forwardkinematics import ForwardKinematic
# ==========================================
# DATASET GENERATION[cite: 3]
# ==========================================
print("Generating Dataset with 6-DOF D-H...")
output_file = os.path.join('dataset_raw.csv')

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    '''
    # Build headers: 6 angles + 21 coordinates (x0, y0, z0 ... x6, y6, z6)
    headers = ['theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5']
    for i in range(7):
        headers.extend([f'x{i}', f'y{i}', f'z{i}'])
    writer.writerow(headers)
    '''
    headers = ['theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'x', 'y', 'z']
    writer.writerow(headers)
    
    for _ in range(500000):
        # 1. Generate Random Angles for 6-DOF
        t1 = np.random.uniform(15.0, 165.0)   # Base
        t2 = np.random.uniform(15.0, 115.0)   # Shoulder
        t3 = np.random.uniform(-90.0, 0.0)    # Elbow
        t4 = np.random.uniform(-90.0, 0.0)    # Wrist Pitch
        t5 = np.random.uniform(-90.0, 90.0)   # Wrist Roll
        t6 = np.random.uniform(0.0, 90.0) # Wrist Twist

        th0 = np.deg2rad(t1)
        th1 = np.deg2rad(t2)
        th2 = np.deg2rad(t3)
        th3 = np.deg2rad(t4)
        th4 = np.deg2rad(t5)
        th5 = np.deg2rad(t6)
        '''
        # 2. Calculate Position
        # Using ForwardKinematic_points for the 21-feature dataset (Type 2)
        xyz_all = ForwardKinematic_points(th0, th1, th2, th3, th4, th5)

        # 3. Save Hardware Angles + Coordinates
        writer.writerow([th0, th1, th2, th3, th4, th5] + list(xyz_all))
        '''
        xyz = ForwardKinematic(th0, th1, th2, th3, th4, th5)
        writer.writerow([th0, th1, th2, th3, th4, th5] + list(xyz))
        

print(f"Dataset saved to {output_file}")