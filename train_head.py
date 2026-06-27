"""Stage 2 (new): train the MLP regression head on cached backbone features.

Replaces ``f2p_trainer.py``. Reuses ``FeatureDataset`` (cached ``features/<name>.npy``
+ ``labels.json``) and the generic ``MLP`` as the head. Config-driven, auto-resumes
from the latest checkpoint, logs to TensorBoard.

Usage:
    .venv/Scripts/python.exe train_head.py --config dinov2_vits14
    .venv/Scripts/python.exe train_head.py --config smoke
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re

import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from src.dataset.datasets import FeatureDataset
from src.models.MLP.MLP import MLP


def set_seed(seed: int) -> int:
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


class HeadTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        os.makedirs(cfg.tb_log_dir, exist_ok=True)
        self.tb = SummaryWriter(log_dir=cfg.tb_log_dir)

        # Train mixes domains (aug_prob); val is deterministic on the in-game features.
        self.train_set = FeatureDataset(cfg.data_dir, is_train=True,
                                        aug_prob=cfg.aug_prob, seed=cfg.seed)
        self.val_set = FeatureDataset(cfg.data_dir, is_train=False,
                                      aug_prob=0.0, seed=cfg.seed)
        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=cfg.num_workers > 0)
        self.val_loader = DataLoader(
            self.val_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=cfg.num_workers > 0)

        self.model = MLP(cfg.feature_dim, cfg.out_dim, cfg.hidden_dim,
                         cfg.num_layers).to(self.device)
        self.lossfunc = MSELoss(reduction="mean").to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, cfg.num_epoch)

        if not self._try_resume():
            self.epoch = 0
            self.global_step = 0
            self.global_val_step = 0

    # --- loss: split MSE on base (shapeValueFace) vs bone params, matching the old convention ---
    def _loss(self, output, labels):
        b = self.cfg.base_dim
        return self.lossfunc(output[:, :b], labels[:, :b]) + \
            self.lossfunc(output[:, b:], labels[:, b:])

    # --- checkpoints ---
    def _try_resume(self) -> bool:
        ckpts = glob.glob(os.path.join(self.cfg.ckpt_dir, "*.pth"))
        if not ckpts:
            print("[train_head] no checkpoint, starting fresh")
            return False

        def epoch_of(s):
            m = re.search(r"ckpt_epoch_(\d+)_step_(\d+).pth", os.path.basename(s))
            return int(m.group(1)) if m else -1

        ckpts.sort(key=epoch_of, reverse=True)
        ckpt = torch.load(ckpts[0], map_location=self.device)
        print(self.model.load_state_dict(ckpt["weights"], strict=False))
        self.optim.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["step"]
        self.global_val_step = ckpt["val_step"]
        print(f"[train_head] resumed from epoch {self.epoch}, step {self.global_step}")
        return True

    def _save(self):
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        os.makedirs(self.cfg.weights_dir, exist_ok=True)
        tag = f"epoch_{self.epoch}_step_{self.global_step}"
        torch.save({
            "exp_name": self.cfg.exp_name, "epoch": self.epoch,
            "step": self.global_step, "val_step": self.global_val_step,
            "weights": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }, os.path.join(self.cfg.ckpt_dir, f"ckpt_{tag}.pth"))
        torch.save({
            "exp_name": self.cfg.exp_name, "epoch": self.epoch,
            "step": self.global_step, "weights": self.model.state_dict(),
        }, os.path.join(self.cfg.weights_dir, f"head_{tag}.pth"))

    # --- loops ---
    def train(self):
        set_seed(self.cfg.seed)
        for _ in range(self.epoch, self.cfg.num_epoch):
            self.epoch += 1
            self.model.train()
            for data in self.train_loader:
                self.global_step += 1
                feat = data["feat"].to(self.device)
                labels = data["label"].to(self.device)
                self.optim.zero_grad()
                output = self.model(feat)
                loss = self._loss(output, labels)
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    sim = torch.nn.functional.cosine_similarity(
                        output, labels, dim=1).mean()
                    dist = torch.nn.functional.pairwise_distance(
                        output, labels, p=2).mean()
                lr = self.optim.param_groups[0]["lr"]
                self.tb.add_scalar("loss", loss, self.global_step)
                self.tb.add_scalar("lr", lr, self.global_step)
                self.tb.add_scalar("cosine_similarity", sim, self.global_step)
                self.tb.add_scalar("distance", dist, self.global_step)
                print(f"epoch {self.epoch} | step {self.global_step} | "
                      f"loss {loss:.6f} | dist {dist:.3f} | cos {sim:.3f}")

            self.scheduler.step()
            if self.epoch % self.cfg.val_interval == 0:
                self.val()
            if self.epoch % self.cfg.ckpt_save_interval == 0:
                self._save()
        self._save()
        self.tb.close()

    @torch.no_grad()
    def val(self):
        self.global_val_step += 1
        self.model.eval()
        tot_loss = tot_sim = tot_dist = 0.0
        n = max(len(self.val_loader), 1)
        for data in self.val_loader:
            feat = data["feat"].to(self.device)
            labels = data["label"].to(self.device)
            output = self.model(feat)
            tot_loss += self._loss(output, labels).item()
            tot_sim += torch.nn.functional.cosine_similarity(
                output, labels, dim=1).mean().item()
            tot_dist += torch.nn.functional.pairwise_distance(
                output, labels, p=2).mean().item()
        self.tb.add_scalar("val/loss", tot_loss / n, self.global_val_step)
        self.tb.add_scalar("val/cosine_similarity", tot_sim / n, self.global_val_step)
        self.tb.add_scalar("val/distance", tot_dist / n, self.global_val_step)
        print(f"[val] step {self.global_val_step} | loss {tot_loss / n:.6f} | "
              f"dist {tot_dist / n:.3f} | cos {tot_sim / n:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    args = ap.parse_args()
    HeadTrainer(get_config(args.config)).train()


if __name__ == "__main__":
    main()
