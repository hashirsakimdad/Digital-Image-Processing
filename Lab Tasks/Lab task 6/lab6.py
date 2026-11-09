import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data

cam = cv2.imread("img/u_proctor.png", cv2.IMREAD_GRAYSCALE)
low = (cam.astype(np.float32) * 0.30 + 85).astype(np.uint8)  # hazy, low contrast version
cv2.imwrite("img/low_contrast.png", low)

out = open("results.txt", "w")
def log(*a):
    print(*a); print(*a, file=out)

# ---------------- Task 1: bit plane extraction ----------------
img = cv2.imread("img/u_group.png", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(6, 4)); plt.imshow(img, cmap="gray"); plt.title("Original Grayscale Image"); plt.axis("off")
plt.savefig("fig_t1_original.png", bbox_inches="tight", dpi=110); plt.close()

planes = []
for k in range(8):
    plane = (img >> k) & 1          # 0 or 1 values
    planes.append(plane)
    log(f"Bit plane {k}: unique values = {np.unique(plane)}, ones = {plane.mean()*100:.1f}%")

fig, ax = plt.subplots(2, 4, figsize=(14, 5.6))
for k in range(8):
    a = ax[k // 4, k % 4]
    a.imshow(planes[k], cmap="gray", vmin=0, vmax=1)
    a.set_title(f"Bit Plane {k}" + (" (LSB)" if k == 0 else " (MSB)" if k == 7 else ""))
    a.axis("off")
plt.tight_layout(); plt.savefig("fig_t1_bitplanes.png", dpi=110); plt.close()

# ---------------- Task 3: histograms of two images ----------------
dark = cv2.imread("img/u_group.png", cv2.IMREAD_GRAYSCALE)
bright = cv2.imread("img/u_code.png", cv2.IMREAD_GRAYSCALE)
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].imshow(dark, cmap="gray", vmin=0, vmax=255); ax[0, 0].set_title("Image A (group photo)"); ax[0, 0].axis("off")
ax[0, 1].imshow(bright, cmap="gray", vmin=0, vmax=255); ax[0, 1].set_title("Image B (code screenshot)"); ax[0, 1].axis("off")
ax[1, 0].hist(dark.ravel(), bins=256, range=(0, 256), color="black"); ax[1, 0].set_title("Histogram of Image A")
ax[1, 1].hist(bright.ravel(), bins=256, range=(0, 256), color="black"); ax[1, 1].set_title("Histogram of Image B")
for a in ax[1]: a.set_xlabel("Intensity"); a.set_ylabel("Pixel count")
plt.tight_layout(); plt.savefig("fig_t3_hist.png", dpi=110); plt.close()
log(f"Image A mean = {dark.mean():.2f}, median = {np.median(dark):.0f}")
log(f"Image B mean = {bright.mean():.2f}, median = {np.median(bright):.0f}")

# ---------------- Task 4: normalized histogram ----------------
img4 = cv2.imread("img/u_proctor.png", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img4], [0], None, [256], [0, 256]).ravel()
total = img4.shape[0] * img4.shape[1]
norm = hist / total
log(f"Total pixels = {total}, sum of hist = {hist.sum():.0f}")
log(f"Sum of normalized histogram = {norm.sum():.6f}")
log(f"Max probability = {norm.max():.5f} at intensity {norm.argmax()}")
plt.figure(figsize=(8, 3.5)); plt.bar(range(256), norm, width=1, color="black")
plt.title("Normalized Histogram (webcam image)"); plt.xlabel("Intensity"); plt.ylabel("p(r)")
plt.tight_layout(); plt.savefig("fig_t4_norm.png", dpi=110); plt.close()

# ---------------- Task 5: contrast stretching ----------------
low = cv2.imread("img/low_contrast.png", cv2.IMREAD_GRAYSCALE)
rmin, rmax = int(low.min()), int(low.max())
stretched = ((low.astype(np.float32) - rmin) / (rmax - rmin) * 255).astype(np.uint8)
log(f"Low contrast image: min = {rmin}, max = {rmax}, std = {low.std():.2f}")
log(f"Stretched image: min = {stretched.min()}, max = {stretched.max()}, std = {stretched.std():.2f}")
log(f"Distinct levels before = {len(np.unique(low))}, after stretching = {len(np.unique(stretched))}")

def compare(imgs, titles, fname):
    fig, ax = plt.subplots(2, len(imgs), figsize=(5.5 * len(imgs), 7))
    for i, (im, t) in enumerate(zip(imgs, titles)):
        ax[0, i].imshow(im, cmap="gray", vmin=0, vmax=255); ax[0, i].set_title(t); ax[0, i].axis("off")
        ax[1, i].hist(im.ravel(), bins=256, range=(0, 256), color="black"); ax[1, i].set_title("Histogram: " + t)
        ax[1, i].set_xlim(0, 255); ax[1, i].set_xlabel("Intensity")
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close()

compare([low, stretched], ["Original (low contrast)", "Contrast Stretched"], "fig_t5_stretch.png")

# ---------------- Task 6: histogram equalization ----------------
equalized = cv2.equalizeHist(low)
log(f"Equalized image: min = {equalized.min()}, max = {equalized.max()}, std = {equalized.std():.2f}")
log(f"Distinct levels after equalization = {len(np.unique(equalized))}")
compare([low, equalized], ["Original (low contrast)", "Histogram Equalized"], "fig_t6_equalize.png")

# ---------------- Task 7: side by side ----------------
compare([low, stretched, equalized], ["Original", "Contrast Stretching", "Histogram Equalization"], "fig_t7_compare.png")
cdf = np.cumsum(cv2.calcHist([equalized], [0], None, [256], [0, 256]).ravel()) / total
plt.figure(figsize=(7, 3.5))
for im, lab in [(low, "Original"), (stretched, "Stretched"), (equalized, "Equalized")]:
    c = np.cumsum(cv2.calcHist([im], [0], None, [256], [0, 256]).ravel()) / total
    plt.plot(c, label=lab)
plt.title("Cumulative distribution (CDF) of each image"); plt.xlabel("Intensity"); plt.ylabel("CDF"); plt.legend()
plt.tight_layout(); plt.savefig("fig_t7_cdf.png", dpi=110); plt.close()
out.close()
