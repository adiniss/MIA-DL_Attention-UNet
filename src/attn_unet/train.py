import torch
from torch.utils.data import DataLoader
from attn_unet.data.stacks2d import Stacks2D
from attn_unet.models.atten_unet import AttentionUNet
from attn_unet.losses.dicece import DiceCE
from attn_unet.losses.focaldice import FocalDice


def batch_dice_from_logits(logits, labels, eps=1e-6):
    # logits: [B,1,H,W] expected; labels: [B,1,H,W] or [B,H,W]
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)                # [B,1,H,W]
    elif labels.ndim != 4:
        raise ValueError(f"labels ndim={labels.ndim}, expected 3 or 4")

    if logits.ndim != 4:
        raise ValueError(f"logits ndim={logits.ndim}, expected 4")

    if logits.shape[1] != 1:
        raise ValueError(f"logits C={logits.shape[1]}, expected 1")

    if labels.shape[1] != 1:
        labels = labels[:, :1]                      # keep first channel if needed

    labels = labels.float()
    probas = torch.sigmoid(logits)
    predictions = (probas > 0.5).float()

    dice_num = (predictions * labels).sum(dim=(1, 2, 3))
    dice_den = predictions.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))
    return ((2.0 * dice_num + eps) / (dice_den + eps)).mean().item()


def main():
    # Run definitions:
    images_dir = globals().get("images_dir",
                               r"/content/drive/MyDrive/MIL-DL_Attention-UNet/Task09_Spleen/imagesTr")
    labels_dir = globals().get("labels_dir",
                               r"/content/drive/MyDrive/MIL-DL_Attention-UNet/Task09_Spleen/labelsTr")

    # Set model parameters
    k_slices = globals().get("k_slices", 1)
    heads = globals().get("heads", 1)
    batch_size = globals().get("batch_size", 4)
    split_ratio = globals().get("split_ratio", (0.8, 0.1, 0.1))
    max_epochs = globals().get("epochs", 6)
    learning_rate = globals().get("learning_rate", 1e-3)
    weight_decay = globals().get("weight_decay", 1e-4)  # L2 reg
    seed = globals().get("seed", 3360033)

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 2

    # Get dataset
    train_ds = Stacks2D(images_dir, labels_dir,
                        k_slices=k_slices,
                        split='train', split_ratio=split_ratio,
                        seed=seed, augment=True)
    val_ds = Stacks2D(images_dir, labels_dir,
                      k_slices=k_slices,
                      split='val', split_ratio=split_ratio,
                      seed=seed, augment=False)

    if len(train_ds) == 0:
        raise RuntimeError("Empty train dataset after split. Check paths/split_ratio.")

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=(device=="cuda"),
                       persistent_workers=True, prefetch_factor=2)

    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=(device=="cuda"),
                       persistent_workers=True, prefetch_factor=2)

    # Set Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionUNet(in_channels=k_slices, num_heads=heads).to(device)

    # Set Loss
    # loss_fn = DiceCE()
    # Testing Focal dice, test alpha in range 0.6-0.9 todo
    loss_fn = FocalDice(alpha=0.75, gamma=2.0, dice_weight=1.0, focal_weight=1.0)

    # Set Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)

    # Initialize best loss, epoch and dice
    best_loss, best_epoch, best_dice = float("inf"), -1, 0.0

    # start training
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    for epoch in range(max_epochs):
        model.train()
        tr_loss_sum = 0.0
        n = 0
        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                logits = model(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss_sum += loss.item() * X.size(0)
            n += X.size(0)
        tr_loss = tr_loss_sum / max(n, 1)

        # validate
        model.eval()
        va_loss_sum = 0.0;
        va_dice_sum = 0.0;
        n = 0
        with torch.no_grad():
            for X, y in dl_va:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                va_loss_sum += loss_fn(logits, y).item() * X.size(0)
                va_dice_sum += batch_dice_from_logits(logits, y) * X.size(0)
                n += X.size(0)
        va_loss = va_loss_sum / max(n, 1)
        va_dice = va_dice_sum / max(n, 1)

        sched.step(va_loss)
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch} | train {tr_loss:.4f} | val {va_loss:.4f} | dice {va_dice:.4f} | l_r = {current_lr:.3e}")

        if va_loss < best_loss:
            best_loss = va_loss
            torch.save(model.state_dict(), "best_loss.pt")

    params = sum(p.numel() for p in model.parameters())
    print(f"Done. Best val loss: {best_loss:.4f} | Params: {params/1e6:.2f}M")

    return {"best_val_loss": best_loss, "best_val_dice": best_dice,
            "best_epoch": best_epoch, "params": params}

if __name__ == "__main__":
    main()
