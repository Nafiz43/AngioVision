import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/30_4_16_32/30_4_16_32_loss.csv"
df = pd.read_csv(csv_path)

# Aggregate per epoch (weighted by batch_size for accuracy)
def weighted_mean(g, col):
    return (g[col] * g["batch_size"]).sum() / g["batch_size"].sum()

per_epoch = df.groupby("epoch").apply(lambda g: pd.Series({
    "loss":      weighted_mean(g, "loss"),
    "clip_loss": weighted_mean(g, "clip_loss"),
    "gen_loss":  weighted_mean(g, "gen_loss"),
    "loss_ema":  g["loss_ema"].iloc[-1],   # EMA: take last value of the epoch
})).reset_index()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(per_epoch["epoch"], per_epoch["loss"],      marker="o", label="Total loss")
ax.plot(per_epoch["epoch"], per_epoch["loss_ema"],  marker="s", label="Loss EMA", linestyle="--")
ax.plot(per_epoch["epoch"], per_epoch["clip_loss"], marker="^", label="CLIP loss")
ax.plot(per_epoch["epoch"], per_epoch["gen_loss"],  marker="d", label="Gen loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Per-epoch training losses (30_4_16_32)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = "/data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/30_4_16_32/30_4_16_32_loss_per_epoch.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved: {out_path}")
print(per_epoch)