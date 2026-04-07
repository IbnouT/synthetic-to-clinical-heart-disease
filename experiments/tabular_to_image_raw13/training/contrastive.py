"""Contrastive SSL pretraining: SimCLR and MoCo.

Both methods pretrain the DeeperCNN backbone on unlabeled images using
the NT-Xent contrastive loss. After pretraining, the backbone is
finetuned with labels using the standard supervised loop.
"""

import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import train_loop
from ..models.expansion import ExpansionLayer, ExpansionModel
from ..config import BATCH_SIZE


def _augment(x):
    """Two augmented views: noise injection and random feature masking."""
    v1 = x + 0.08 * torch.randn_like(x)
    v2 = x * (torch.rand_like(x) > 0.12).float() + 0.03 * torch.randn_like(x)
    return v1, v2


def _nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent contrastive loss (same as SimCLR/SCARF)."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    B = z1.shape[0]
    z = torch.cat([z1, z2])
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    idx = torch.arange(B, device=z.device)
    return F.cross_entropy(sim, torch.cat([idx + B, idx]))


def train_simclr(backbone, X_all, y_all, X_train, y_train, X_val, y_val,
                 device, config):
    """SimCLR: contrastive pretrain -> finetune with labels.

    Returns (model, pretrain_state, comp_auc, comp_preds, elapsed).
    """
    expansion = ExpansionLayer().to(device)
    backbone = backbone.to(device)
    # Projection head for contrastive space
    proj = nn.Sequential(
        nn.Linear(backbone.REP_DIM, 128), nn.BatchNorm1d(128), nn.GELU(),
        nn.Linear(128, 64)).to(device)

    opt = torch.optim.AdamW(
        list(expansion.parameters()) + list(backbone.parameters()) + list(proj.parameters()),
        lr=config["pretrain_lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["pretrain_epochs"])

    loader = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"    SimCLR pretraining ({config['pretrain_epochs']} epochs)...")
    t0 = time.time()

    for ep in range(config["pretrain_epochs"]):
        expansion.train(); backbone.train(); proj.train()
        total, n = 0, 0
        for (bx,) in loader:
            bx = bx.to(device)
            img = expansion(bx)
            v1, v2 = _augment(img)
            loss = _nt_xent_loss(proj(backbone.extract(v1)), proj(backbone.extract(v2)))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        sch.step()
        if (ep + 1) % 20 == 0:
            print(f"      Epoch {ep+1}/{config['pretrain_epochs']}: loss={total/n:.4f}")

    # Save pretrained state (before finetuning)
    pretrain_state = {
        "expansion": {k: v.cpu().clone() for k, v in expansion.state_dict().items()},
        "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
    }

    # Finetune with labels
    print(f"    Finetuning ({config['finetune_epochs']} epochs)...")
    model = ExpansionModel(backbone)
    # Load pretrained expansion weights
    model.expansion.load_state_dict(pretrain_state["expansion"])
    model.backbone.load_state_dict(pretrain_state["backbone"])

    auc, preds, _ = train_loop(
        model, X_train, y_train, X_val, y_val, device,
        epochs=config["finetune_epochs"], lr=config["finetune_lr"])

    elapsed = time.time() - t0
    print(f"    Best AUC: {auc:.6f} ({elapsed/60:.1f} min total)")

    return model, pretrain_state, auc, preds, elapsed


def train_moco(backbone_factory, X_all, y_all, X_train, y_train, X_val, y_val,
               device, config):
    """MoCo: momentum contrastive pretrain -> finetune with labels.

    Uses a momentum-updated key encoder and a queue of past negatives.
    Returns (model, pretrain_state, comp_auc, comp_preds, elapsed).
    """
    expansion = ExpansionLayer().to(device)
    backbone = backbone_factory().to(device)
    key_enc = deepcopy(backbone).to(device)
    for p in key_enc.parameters():
        p.requires_grad = False

    q_proj = nn.Sequential(
        nn.Linear(backbone.REP_DIM, 128), nn.BatchNorm1d(128), nn.GELU(),
        nn.Linear(128, 64)).to(device)
    k_proj = deepcopy(q_proj).to(device)
    for p in k_proj.parameters():
        p.requires_grad = False

    queue_size = config["queue_size"]
    queue = F.normalize(torch.randn(64, queue_size, device=device), dim=0)
    queue_ptr = 0
    mom = config["momentum"]

    opt = torch.optim.AdamW(
        list(expansion.parameters()) + list(backbone.parameters()) + list(q_proj.parameters()),
        lr=config["pretrain_lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["pretrain_epochs"])

    loader = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"    MoCo pretraining ({config['pretrain_epochs']} epochs)...")
    t0 = time.time()

    for ep in range(config["pretrain_epochs"]):
        expansion.train(); backbone.train()
        total, n = 0, 0
        for (bx,) in loader:
            bx = bx.to(device)
            img = expansion(bx)
            v1, v2 = _augment(img)
            q = F.normalize(q_proj(backbone.extract(v1)), dim=1)
            with torch.no_grad():
                k = F.normalize(k_proj(key_enc.extract(v2)), dim=1)
            # Positive and negative logits
            l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', q, queue.clone().detach())
            logits = torch.cat([l_pos, l_neg], dim=1) / 0.07
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()

            # Momentum update of key encoder and projection
            with torch.no_grad():
                for qp, kp in zip(backbone.parameters(), key_enc.parameters()):
                    kp.data = mom * kp.data + (1 - mom) * qp.data
                for qp, kp in zip(q_proj.parameters(), k_proj.parameters()):
                    kp.data = mom * kp.data + (1 - mom) * qp.data
                # Enqueue current keys
                bk = k.T
                bs = bk.shape[1]
                end = min(queue_ptr + bs, queue_size)
                queue[:, queue_ptr:end] = bk[:, :end - queue_ptr]
                queue_ptr = end % queue_size

            total += loss.item(); n += 1
        sch.step()
        if (ep + 1) % 20 == 0:
            print(f"      Epoch {ep+1}/{config['pretrain_epochs']}: loss={total/n:.4f}")

    pretrain_state = {
        "expansion": {k: v.cpu().clone() for k, v in expansion.state_dict().items()},
        "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
    }

    # Finetune
    print(f"    Finetuning ({config['finetune_epochs']} epochs)...")
    model = ExpansionModel(backbone)
    model.expansion.load_state_dict(pretrain_state["expansion"])
    model.backbone.load_state_dict(pretrain_state["backbone"])

    auc, preds, _ = train_loop(
        model, X_train, y_train, X_val, y_val, device,
        epochs=config["finetune_epochs"], lr=config["finetune_lr"])

    elapsed = time.time() - t0
    print(f"    Best AUC: {auc:.6f} ({elapsed/60:.1f} min total)")

    return model, pretrain_state, auc, preds, elapsed
