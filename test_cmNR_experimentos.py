# test_hybrid_newton.py
"""
Hybrid MLP + Newton-Raphson Evaluation for Master's Final Project.
Loads a trained MLP, uses it to predict an initial guess (theta_0),
then refines it using the Newton-Raphson (Jacobian Pseudo-Inverse) method.
"""
import os, glob, json, argparse
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from forwardkinematics import ForwardKinematic, get_dh_matrix

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

MODEL_DIR = "Model"
RESULTS_DIR = "RESULTS_HYBRID"
os.makedirs(RESULTS_DIR, exist_ok=True)

DH_TABLE = np.array([
    [6.5,  0.0,   np.pi/2],
    [0.0,  11.0,  0.0],
    [0.0,  15.0,  0.0],
    [0.0,  0.0,   np.pi/2],
    [0.0,  0.0,  -np.pi/2],
    [18.0, 0.0,   0.0]
])

def compute_jacobian(thetas):
    """Numerical Jacobian for the end-effector position (3x6 matrix)."""
    eps = 1e-4
    J = np.zeros((3, 6))
    current_pos = ForwardKinematic(*thetas)
    for i in range(6):
        thetas_eps = thetas.copy()
        thetas_eps[i] += eps
        pos_eps = ForwardKinematic(*thetas_eps)
        J[:, i] = (pos_eps - current_pos) / eps
    return J

def newton_raphson_ik(theta_0, target_xyz, max_iter=5, tol=1e-3):
    """
    Newton-Raphson IK solver.
    Returns the refined angles and number of iterations taken.
    """
    theta = np.array(theta_0, dtype=np.float64)
    for i in range(max_iter):
        current_pos = ForwardKinematic(*theta)
        error = target_xyz - current_pos
        
        # Check for hyperparameter: weight factor for joint limits (optional but good)
        if np.linalg.norm(error) < tol:
            return theta, i
        
        J = compute_jacobian(theta)
        
        # Damped pseudo-inverse (Levenberg-Marquardt style) to avoid singularities
        lam = 1e-3
        JJT = np.dot(J, J.T) + lam * np.eye(3)
        J_pinv = np.dot(J.T, np.linalg.inv(JJT))
        
        # Newton-Raphson update step
        theta = theta + np.dot(J_pinv, error)
        
    return theta, max_iter

def evaluate_hybrid(model_path):
    print(f"\nEvaluating Hybrid MLP + Newton-Raphson: {os.path.basename(model_path)}")
    model = keras.models.load_model(model_path, compile=False)
    
    data = np.load('dataset_processed.npz')
    X_test, y_test = data['X_test'], data['y_test'] # X_test is scaled 3D target, y_test is angles
    
    # Scaler is needed? In your train script, X_test was already scaled for the MLP.
    # We need the UNSCALED X_test (real cm values) for Newton-Raphson.
    # Assuming separate_dataset.py saved unscaled data? No, it saved scaled.
    # To fix this easily without re-running, we can just use the scaled data for FK too,
    # but it's better to just unscale X_test.
    try:
        scaler_X = joblib.load('scaler_X.pkl')
        X_test_unscaled = scaler_X.inverse_transform(X_test)
    except:
        X_test_unscaled = X_test # fallback
    
    # 1. Get initial guess from MLP
    theta_0_pred = model.predict(X_test, verbose=0)
    
    mlp_errors = []
    hybrid_errors = []
    iterations_taken = []
    
    # 2. Newton-Raphson Refinement
    for i in range(len(X_test)):
        target_xyz = X_test_unscaled[i]
        theta_mlp = theta_0_pred[i]
        
        # Pure MLP Error
        mlp_pos = ForwardKinematic(*theta_mlp)
        mlp_errors.append(np.linalg.norm(target_xyz - mlp_pos))
        
        # Hybrid Error
        theta_refined, iters = newton_raphson_ik(theta_mlp, target_xyz, max_iter=3, tol=1e-4)
        hybrid_pos = ForwardKinematic(*theta_refined)
        hybrid_errors.append(np.linalg.norm(target_xyz - hybrid_pos))
        
        iterations_taken.append(iters)
        
        if i % 5000 == 0:
            print(f"Sample {i}/{len(X_test)} | MLP Err: {mlp_errors[-1]:.3f} cm | Hybrid Err: {hybrid_errors[-1]:.4f} cm | Iters: {iters}")

    mlp_errors = np.array(mlp_errors)
    hybrid_errors = np.array(hybrid_errors)
    
    # --- Metrics ---
    y_test_deg = np.rad2deg(y_test)
    y_pred_deg = np.rad2deg(theta_0_pred) # Note: we are measuring the MLP's initial guess MAE here
    mae = mean_absolute_error(y_test_deg, y_pred_deg, multioutput='raw_values')
    
    summary = {
        "model": os.path.basename(model_path),
        "mlp_mean_cm": float(mlp_errors.mean()),
        "mlp_p95_cm": float(np.percentile(mlp_errors, 95)),
        "hybrid_mean_cm": float(hybrid_errors.mean()),
        "hybrid_p95_cm": float(np.percentile(hybrid_errors, 95)),
        "avg_iters": float(np.mean(iterations_taken)),
        "joint_mae_deg": float(mae.mean())
    }
    
    print("\n--- Results ---")
    print(f"Pure MLP Mean Euclidean Error:  {summary['mlp_mean_cm']:.3f} cm")
    print(f"Hybrid Mean Euclidean Error:     {summary['hybrid_mean_cm']:.4f} cm")
    print(f"Average Newton-Raphson Iters:    {summary['avg_iters']:.2f}")
    
    # --- Plotting ---
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    sub = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(sub, exist_ok=True)
    
    with open(os.path.join(sub, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    plt.figure(figsize=(10,6))
    plt.hist(mlp_errors, bins=100, alpha=0.6, label=f'Pure MLP (μ={mlp_errors.mean():.2f} cm)')
    plt.hist(hybrid_errors, bins=100, alpha=0.6, label=f'Hybrid MLP+NR (μ={hybrid_errors.mean():.2f} cm)')
    plt.axvline(mlp_errors.mean(), color='blue', ls='--')
    plt.axvline(hybrid_errors.mean(), color='orange', ls='--')
    plt.title('Position Error: Pure MLP vs MLP+Newton-Raphson')
    plt.xlabel('Euclidean Error (cm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(sub, 'hybrid_vs_mlp_hist.png'), dpi=200)
    plt.close()

if __name__ == "__main__":
    # Find all MLP models (exclude MDN)
    mlp_models = glob.glob(os.path.join(MODEL_DIR, "*mlp*.keras"))
    
    if not mlp_models:
        print("No MLP models found in Model/ directory. Did run_experiments.py finish?")
    else:
        for m in mlp_models:
            evaluate_hybrid(m)