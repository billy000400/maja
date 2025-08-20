import os
import math
import torch
from torch.utils.data import DataLoader, DistributedSampler
from yourproj.data.datasets import H5LazyDataset
from yourproj.models.mlp import MLPRegressor
from yourproj.logging.logger import Logger
from yourproj.engine.trainer import Trainer

# (parse args or load YAML here)
train_h5 = "combined_train.h5"
features = ["FEATURES/x", "FEATURES/y"]
label = "TARGETS/gen_mass"

train_ds = H5LazyDataset(train_h5, features, label)
n = len(train_ds)
val_frac = 1 - 90/95  # assume 90:5:5 split
n_val = int(math.ceil(val_frac*n))
idx = torch.randperm(n).numpy()
val_idx, tr_idx = idx[:n_val], idx[n_val:]


class Subset(torch.utils.data.Dataset):
    def __init__(self, base, ids): self.base, self.ids = base, ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.base[int(self.ids[i])]


train_dl = DataLoader(Subset(train_ds, tr_idx), batch_size=1024, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
val_dl = DataLoader(Subset(train_ds, val_idx), batch_size=1024, num_workers=2, shuffle=False, pin_memory=True)

# build model
sample_x, _ = train_ds[0]
in_dim = int(sample_x.numel())
model = MLPRegressor(in_dim, hidden=512, depth=3, dropout=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler()
logger = Logger(log_dir="outputs/tb/exp1", use_wandb=True, wandb_cfg={"project": "genmass", "name": "exp1"})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(model, opt, scaler, logger, device, ddp=False)


def save_ckpt(m, epoch, best):
    state = m.module.state_dict() if hasattr(m, "module") else m.state_dict()
    os.makedirs("outputs/ckpts/exp1", exist_ok=True)
    torch.save({"epoch": epoch, "state": state, "best": best}, f"outputs/ckpts/exp1/best.pt")

trainer.fit(train_dl, val_dl, epochs=20, log_every=50, ckpt_fn=save_ckpt)
logger.close()

