import os, torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from attn_unet.data.stacks2d import Stacks2D
from attn_unet.models.atten_unet import AttentionUNet
from attn_unet.losses.dicece import DiceCE


def batch_dice_from_logits(logits, labels, eps=1e-6):
    # make sure it's the expected shape (N, 1, H, W)
    assert labels.ndim == 4 and labels.shape[1] == 1
    labels = labels.float()

    assert logits.shape[1] == 1
    probas = torch.sigmoid(logits)
    predictions = (probas > 0.5).float()

    # todo verify dice
    dice_num = (predictions * labels).sum(dim=(1, 2, 3))
    dice_den = predictions.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))
    return ((2.0 * dice_num + eps) / (dice_den + eps)).mean().item()


def main():
    # Run definitions:
    images_dir = globals().get("images_dir",
                               r"/content/drive/MyDrive/MIL-DL_Attention-UNet/Task07_Pancreas_small/imagesTr")
    labels_dir = globals().get("labels_dir",
                               r"/content/drive/MyDrive/MIL-DL_Attention-UNet/Task07_Pancreas_small/labelsTr")

    # Set model parameters
    k_slices = globals().get("k_slices", 1)
    heads = globals().get("heads", 1)
    batch_size = globals().get("batch_size", 4)
    max_epochs = globals().get("epochs", 12)
    patience = globals().get("patience", 2)  # for scheduler
    learning_rate = globals().get("learning_rate", 1e-3)
    weight_decay = globals().get("weight_decay", 1e-4)  # L2 reg

    # Get dataset
    train_ds = Stacks2D(images_dir, labels_dir, k_slices=k_slices, split='train')
    val_ds = Stacks2D(images_dir, labels_dir, k_slices=k_slices, split='val')

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # set model, optimizer scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionUNet(in_channels=k_slices, num_heads=heads).to(device)
    loss_fn = DiceCE()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)

    # Initialize best loss, epoch and dice
    best_loss, best_epoch, best_dice = float("inf"), -1, 0.0

    print(f"train: {len(train_ds)} slices")
    print(f"val: {len(val_ds)} slices")

    # start training
    for epoch in range(max_epochs):
        model.train();
        tr_loss = 0.0
        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(len(dl_tr), 1)

        model.eval();
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

        print(f"Epoch {epoch} | train {tr_loss:.3f} | val {va_loss:.3f} | dice {va_dice:.3f}")

        sched.step(va_loss)
        print("current LR:", sched.get_last_lr()[0])
        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch, best_dice = va_loss, epoch, va_dice
        if epoch - best_epoch >= patience:
            print(f"Early stop @ {epoch} (best {best_loss:.3f} at {best_epoch})")
            break

    params = sum(p.numel() for p in model.parameters())
    return {"best_val_loss": best_loss, "best_val_dice": best_dice,
            "best_epoch": best_epoch, "params": params}


if __name__ == "__main__":
    main()
