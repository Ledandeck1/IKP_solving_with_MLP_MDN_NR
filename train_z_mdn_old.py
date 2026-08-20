from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_mdn_layer as mdn
import os
import matplotlib.pyplot as plt

MODEL_DIR = "Model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Training 6-DOF Inverse Kinematics MDN Model (3 Inputs -> 6 Outputs)...")

# 1. Load Data
data = np.load('dataset_processed.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']

OUTPUT_DIMS = 6
N_MIXES = 10  # Fixed: Kept consistent with test_mdn.py

# 2. Build Model 
model = keras.Sequential([
    layers.Input(shape=(3,)),             # 3 Inputs (Target X, Y, Z)
    #layers.Dense(1024, activation='relu'),
    #layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    #layers.Dense(256, activation='gelu'),
    #layers.Dense(128, activation='gelu'),
    #layers.Dense(64, activation='gelu'),
    #layers.Dense(32, activation='gelu'),
    #layers.Dense(256,activation='tanh'),
    #layers.Dense(128,activation='tanh'),
    #layers.Dense(64,activation='tanh'),    
    #layers.Dense(32,activation='tanh'),
    mdn.MDN(OUTPUT_DIMS, N_MIXES)         # 6 Outputs, 5 Gaussian Mixtures
])

# 3. Optimizer Configuration
base_optimizer = keras.optimizers.Muon(
    learning_rate=0.00001
)

model.compile(
    optimizer=base_optimizer,
    # THIS MUST MATCH the (6, 5) defined in the MDN layer above
    loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES) 
)
model.summary()

# 4. Callbacks (Early Stopping & Learning Rate Reduction)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = os.path.join(MODEL_DIR, f'best_model_{timestamp}_MDN.keras')

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        save_best_only=True, 
        monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=15,
        min_lr=1e-4,
        verbose=1
    )
]

# 5. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    validation_steps=500,
    epochs=1000,
    batch_size=64,  
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=2048
)

# 6. Plotting Validation History
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(loss))

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('MDN Training & Validation Loss (NLL)')
plt.xlabel('Epoch')
plt.ylabel('Negative Log Likelihood')

val_plot_path = os.path.join(MODEL_DIR, f'plot_validation_{timestamp}_MDN.png')
plt.savefig(val_plot_path)
plt.close()

print(f"Validation plot saved to: {val_plot_path}")
print(f"Training complete. Best model saved to: {checkpoint_path}")