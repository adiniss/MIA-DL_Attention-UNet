import os, torch
from torch.utils.data import DataLoader
from attn_unet.data.stacks2d import Stacks2D
from attn_unet.models.atten_unet import AttentionUNet
from attn_unet.losses.dicece import DiceCE


def main():
    # Run definitions:
    images_dir = globals().get("images_dir",
                               r"/content/drive/MyDrive/MIL-DL_Attention-UNet/Task07_Pancreas_small/imagesTr")
    labels_dir = globals().get("labels_dir",
                               r"/content/drive/MyDrive/MIL-DL_Attention-UNet/Task07_Pancreas_small/labelsTr")


    k_slices = globals().get("k_slices", 1)
    heads = globals().get("heads", 1)
    batch_size = globals().get("batch_size", 4)
    max_epochs = globals().get("epochs", 12)
    patience = globals().get("patience", 2)  # for scheduler
    learning_rate = globals().get("learning_rate", 1e-3)
    weight_decay = globals().get("weight_decay", 1e-4)  # L2 reg



    train_ds = Stacks2D(images_dir, labels_dir, k_slices=k_slices, split='train')
    val_ds = Stacks2D(images_dir, labels_dir, k_slices=k_slices, split='val')

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionUNet(in_channels=k_slices, num_heads=heads).to(device)
    loss_fn = DiceCE()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1, verbose=True)

    # Initialize best loss and epoch
    best_loss, best_epoch = float("inf"), -1

    print(f"train: {len(train_ds)} slices")
    print(f"val: {len(val_ds)} slices")

    # start training
    for epoch in range(max_epochs):
        model.train()
        tr_loss = 0.0
        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip grads
            opt.step()
            tr_loss += loss.item()

        tr_loss /= len(dl_tr)

        # evaluate model
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, y in dl_va:
                X, y = X.to(device), y.to(device)
                va_loss += loss_fn(model(X), y).item() * X.size(0)
        va_loss /= len(dl_va.dataset)

        print(f"Epoch {epoch} | train {tr_loss:.3f} | val {va_loss:.3f}")

        # scheduler + checkpoint + early stop
        sched.step(va_loss)
        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch = va_loss, epoch
        if epoch - best_epoch >= patience:
            print(f"Early stop @ {epoch} (best {best_loss:.3f} at {best_epoch})")
            break

if __name__ == "__main__":
    main()
