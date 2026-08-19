# train_z_mdn.py
"""
Unified training script for IEEE Latin America IK study.
Supports:
  --model {mlp, mdn}
  --optimizer {adamw, muon, shampoo}
  --seed <int>
  --run_id <int>     (0..9)
Trains one configuration; saves model + training history JSON.
"""
import os, json, argparse, hashlib, time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_mdn_layer as mdn
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

#parser para o json com os hiperparametros da rede
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["mlp", "mdn"], required=True)
parser.add_argument("--optimizer", choices=["adamw", "muon", "shampoo"], required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--steps_per_epoch", type=int, default=2048)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--n_mixes", type=int, default=20)
parser.add_argument("--patience", type=int, default=20)
args = parser.parse_args()

set_global_seed(args.seed)

MODEL_DIR = "Model"
RUNS_DIR  = "Runs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

#salvando os modelos com print do tempo

print(f"[{datetime.now().isoformat()}] Training "
      f"model={args.model} opt={args.optimizer} seed={args.seed} run_id={args.run_id}")

#Dataset carregado para treinar a rede

data = np.load("dataset_processed.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
INPUT_DIM  = X_train.shape[1]
OUTPUT_DIM = y_train.shape[1]

#arquitetura de input e camadas ocultas a serem utilizadas

def build_hidden_stack(input_dim):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense( 64, activation="relu"),
        layers.Dense( 32, activation="relu"),
    ])

if args.model == "mlp":
    model = build_hidden_stack(INPUT_DIM)
    model.add(layers.Dense(OUTPUT_DIM, activation="linear"))
    loss_fn = "mse"
    metrics = ["mae"]
else:  # mdn
    model = build_hidden_stack(INPUT_DIM)
    model.add(mdn.MDN(OUTPUT_DIM, args.n_mixes))
    loss_fn = mdn.get_mixture_loss_func(OUTPUT_DIM, args.n_mixes)
    metrics = []

# aqui estão os 3 tipos de otimizadores utilizados no experimento

if args.optimizer == "adamw":
    opt = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )
elif args.optimizer == "muon":
    
    if hasattr(keras.optimizers, "Muon"):
        opt = keras.optimizers.Muon(learning_rate=args.lr, momentum=0.95)
    else:
        
        opt = keras.optimizers.AdamW(learning_rate=args.lr*0.1,
                                     weight_decay=args.weight_decay)
elif args.optimizer == "shampoo":
    try:
        import tensorflow_addons as tfa
        opt = tfa.optimizers.Shampoo(learning_rate=args.lr)
    except Exception:
        
        opt = keras.optimizers.Adam(learning_rate=args.lr)

model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
model.summary()

#configuracao do checkpoint para salvar o modelo treinado
config_str = f"{args.model}_{args.optimizer}_seed{args.seed}_run{args.run_id}"
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
tag        = f"{timestamp}_{timestamp}_{config_str}"
checkpoint_path = os.path.join(MODEL_DIR, f"best_model_{config_str}.keras")
history_path   = os.path.join(RUNS_DIR, f"history_{config_str}.json")
config_path    = os.path.join(RUNS_DIR,  f"config_{config_str}.json")

#definição dos callbacks para early stopping, reduceLRonplateau e checkpoint do modelo

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                  patience=args.patience,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                    save_best_only=True,
                                    monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                      factor=0.4, patience=15,
                                      min_lr=1e-5, verbose=1),
]

#treino efetivo

t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    validation_steps=500,
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=args.steps_per_epoch,
)
train_time = time.time() - t0

#Configuração para salvar os dados obtidos como tempo de treino, parada, loss

hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
hist["train_time_sec"] = train_time
hist["stopped_epoch"]  = len(hist["loss"])
with open(history_path, "w") as f:
    json.dump(hist, f, indent=2)
#JSON estruturado para capturar informações sobre o modelo treinado
config_dump = {
    "model":     args.model,
    "optimizer": args.optimizer,
    "seed":      args.seed,
    "run_id":    args.run_id,
    "lr":        args.lr,
    "weight_decay": args.weight_decay,
    "batch_size":   args.batch_size,
    "n_mixes":      args.n_mixes if args.model == "mdn" else None,
    "input_dim":    INPUT_DIM,
    "output_dim":   OUTPUT_DIM,
    "checkpoint":   checkpoint_path,
    "architecture": "256-128-64-32 ReLU",
}
with open(config_path, "w") as f:
    json.dump(config_dump, f, indent=2)

#plot da loss e validation curve

plt.figure(figsize=(8,6))
plt.plot(hist["loss"],     label="Training Loss")
plt.plot(hist["val_loss"], label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
loss_name = "MSE" if args.model == "mlp" else "NLL"
plt.title(f"{args.model.upper()} ({args.optimizer}) — {loss_name}")
plt.legend(loc="upper right")
plt.savefig(os.path.join(RUNS_DIR, f"plot_{config_str}.png"))
plt.close()

print(f"Done. model={checkpoint_path} time={train_time:.1f}s epochs={hist['stopped_epoch']}")