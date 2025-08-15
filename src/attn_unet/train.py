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
    out_dir = r"./runs/debug"
    os.makedirs(out_dir, exist_ok=True)

    # Model definitions:
    k_slices = globals().get("k_slices", 5)
    batch_size = globals().get("batch_size", 2)
    heads = globals().get("heads", 4)
    in_channels = globals().get("in_channels", 5)
    learning_rate = globals().get("learning_rate", 1e-3)

    # Load images as 2D stacks and get the dataloader
    stacks_train = Stacks2D(images_dir, labels_dir, k_slices=k_slices, split='train')
    stacks_val = Stacks2D(images_dir, labels_dir, k_slices=k_slices, split='val')

    data_train = DataLoader(stacks_train, batch_size=batch_size, shuffle=True, num_workers=0)
    data_val = DataLoader(stacks_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # Define model, optimizer (using Adam like the paper) and loss
    model = AttentionUNet(in_channels=in_channels, num_heads=heads).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = DiceCE()

    best_loss = 1e9
    for epoch in range(5):  # small run for this PoC
        model.train(); tr_loss = 0
        for X, y in data_train:
            X, y = X.cuda(), y.cuda()

            # zero gradients
            opt.zero_grad()

            # Get logits from model's forward pass
            logits = model(X)

            # Find loss
            loss = loss_fn(logits, y)

            # Step and log loss
            loss.backward()
            opt.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(data_train.dataset)

        model.eval(); va_loss = 0
        with torch.no_grad():
            for X, y in data_val:
                X, y = X.cuda(), y.cuda()
                va_loss += loss_fn(model(X), y).item() * X.size(0)
        va_loss /= len(data_val.dataset)

        print(f"Epoch {epoch} | train {tr_loss:.3f} | val {va_loss:.3f}")
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_loss.pt"))

if __name__ == "__main__":
    main()
