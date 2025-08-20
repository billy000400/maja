import torch, time
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, model, optimizer, scaler, logger, device, ddp=False, local_rank=0):
        self.model = model.to(device)
        if ddp:
            self.model = DDP(self.model, device_ids=[local_rank])
        self.opt, self.scaler, self.logger, self.device = optimizer, scaler, logger, device

    def fit(self, train_loader, val_loader, epochs, log_every=50, ckpt_fn=None):
        best = float("inf")
        step = 0
        for epoch in range(epochs):
            self.model.train()
            t0 = time.time()
            for bi, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                self.opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(True):
                    pred = self.model(x)
                    loss = self.model.module.loss(pred, y) if hasattr(self.model, "module") else self.model.loss(pred, y)
                self.scaler.scale(loss).step(self.opt)
                self.scaler.update()
                if (bi + 1) % log_every == 0:
                    with torch.no_grad():
                        mae = (pred - y).abs().mean().item()
                    self.logger.log(step, {"loss": loss.item(), "mae": mae, "ips": (bi+1)*x.size(0)/(time.time()-t0+1e-9)}, "train_iter")
                step += 1

            # validation
            val = self.evaluate(val_loader)
            self.logger.log(epoch, val, "val")
            if val["loss"] < best and ckpt_fn:
                best = val["loss"]
                ckpt_fn(self.model, epoch, best)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, total_mae, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            pred = self.model(x)
            if hasattr(self.model, "module"):
                loss = self.model.module.loss(pred, y)
            else:
                loss = self.model.loss(pred, y)
            bs = y.size(0)
            n += bs
            total_loss += loss.item() * bs
            total_mae += (pred - y).abs().mean().item() * bs
        return {"loss": total_loss / max(1, n), "mae": total_mae / max(1, n)}

