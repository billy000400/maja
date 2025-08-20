from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir, use_wandb=False, wandb_cfg=None):
        self.tb = SummaryWriter(log_dir)
        self.wandb = None
        if use_wandb:
            import wandb
            self.wandb = wandb.init(**(wandb_cfg or {}))

    def log(self, step, metrics, prefix="train"):
        for k, v in metrics.items():
            self.tb.add_scalar(f"{prefix}/{k}", v, step)
        if self.wandb:
            self.wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)

    def close(self):
        self.tb.close()
        if self.wandb: self.wandb.finish()

