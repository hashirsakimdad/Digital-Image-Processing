"""
Generate Digital Image Processing lab outputs using the NIC image.
Saves results under Output/Lab N/Task M/
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
NIC_PATH = ROOT / "Lab Tasks" / "Lab task 1" / "nic ront.jpg"
OUT = ROOT / "Output"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(path: Path, tight: bool = True) -> None:
    ensure_dir(path.parent)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_bgr(path: Path, img_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img_bgr)


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_nic() -> np.ndarray:
    if not NIC_PATH.exists():
        raise FileNotFoundError(f"NIC image not found: {NIC_PATH}")
    img = cv2.imread(str(NIC_PATH))
    if img is None:
        raise FileNotFoundError(f"Could not read NIC image: {NIC_PATH}")
    return img


def write_info(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# LAB 1 — Image loading, display, properties, save
# ---------------------------------------------------------------------------
def lab1(img_bgr: np.ndarray) -> None:
    base = ensure_dir(OUT / "Lab 1" / "Task 1")
    img_rgb = to_rgb(img_bgr)
    h, w, c = img_bgr.shape

    # Original display
    plt.figure(figsize=(8, 5))
    plt.imshow(img_rgb)
    plt.title("Lab 1 — Original (OpenCV + Matplotlib)")
    plt.axis("off")
    save_fig(base / "01_original_display.png")

    # Properties
    info = (
        f"Height   : {h}\n"
        f"Width    : {w}\n"
        f"Channels : {c}\n"
        f"Data type: {img_bgr.dtype}\n"
        f"Minimum  : {img_bgr.min()}\n"
        f"Maximum  : {img_bgr.max()}\n"
    )
    row, col = 100, 150
    if h > row and w > col:
        pixel_bgr = img_bgr[row, col]
        pixel_rgb = img_rgb[row, col]
        info += (
            f"\nPixel at ({row}, {col}) BGR : {pixel_bgr}\n"
            f"Pixel at ({row}, {col}) RGB : {pixel_rgb}\n"
        )
    write_info(base / "02_properties.txt", info)

    # Pixel highlight
    marked = img_bgr.copy()
    cv2.circle(marked, (col, row), 12, (0, 0, 255), 2)
    save_bgr(base / "03_pixel_access_marked.jpg", marked)

    # Pillow read / display
    img_pil = Image.open(NIC_PATH)
    plt.figure(figsize=(8, 5))
    plt.imshow(img_pil)
    plt.title("Lab 1 — Read with Pillow")
    plt.axis("off")
    save_fig(base / "04_pillow_display.png")
    write_info(
        base / "04_pillow_info.txt",
        f"Size (W x H): {img_pil.size}\nMode: {img_pil.mode}\nFormat: {img_pil.format}\n",
    )

    # Matplotlib read
    img_mpl = mpimg.imread(str(NIC_PATH))
    plt.figure(figsize=(8, 5))
    plt.imshow(img_mpl)
    plt.title("Lab 1 — Read with Matplotlib")
    plt.axis("off")
    save_fig(base / "05_matplotlib_display.png")
    write_info(
        base / "05_matplotlib_info.txt",
        f"Shape: {img_mpl.shape}\nDtype: {img_mpl.dtype}\n"
        f"Min: {img_mpl.min()}\nMax: {img_mpl.max()}\n",
    )

    # Save with OpenCV & Pillow
    save_bgr(base / "06_opencv_saved.jpg", img_bgr)
    img_pil.save(base / "07_pillow_saved.jpg")
    print("Lab 1 done ->", base)


# ---------------------------------------------------------------------------
# LAB 2 — Crop, resize, flip
# ---------------------------------------------------------------------------
def lab2(img_bgr: np.ndarray) -> None:
    base = ensure_dir(OUT / "Lab 2" / "Task 1")
    img_rgb = to_rgb(img_bgr)
    h, w, c = img_bgr.shape

    plt.figure(figsize=(8, 5))
    plt.imshow(img_rgb)
    plt.title("Lab 2 — Original")
    plt.axis("off")
    save_fig(base / "01_original.png")

    write_info(
        base / "02_properties.txt",
        f"Height: {h}\nWidth: {w}\nChannels: {c}\nDtype: {img_bgr.dtype}\n"
        f"Min: {img_bgr.min()}\nMax: {img_bgr.max()}\n",
    )

    # Access pixel (100, 100)
    r, c0 = 100, 100
    pixel = img_bgr[r, c0]
    write_info(base / "03_pixel_100_100.txt", f"Pixel BGR at (100,100): {pixel}\n")
    marked = img_bgr.copy()
    cv2.circle(marked, (c0, r), 12, (0, 255, 0), 2)
    save_bgr(base / "03_pixel_marked.jpg", marked)

    # BGR -> RGB
    save_bgr(base / "04_bgr.jpg", img_bgr)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_bgr)  # wrong colors intentionally to show BGR
    plt.title("Shown as BGR (incorrect colors)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title("Converted BGR → RGB")
    plt.axis("off")
    save_fig(base / "04_bgr_to_rgb.png")

    # Crop center region (or photo area of NIC — center 50%)
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    cropped = img_bgr[y1:y2, x1:x2].copy()
    save_bgr(base / "05_cropped.jpg", cropped)

    # Resize to 400 x 300 (width x height in OpenCV: (400, 300))
    resized = cv2.resize(cropped, (400, 300), interpolation=cv2.INTER_AREA)
    save_bgr(base / "06_resized_400x300.jpg", resized)

    # Flip horizontally
    flipped = cv2.flip(resized, 1)
    save_bgr(base / "07_flipped_horizontal.jpg", flipped)

    # Display all together
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        (img_rgb, "Original"),
        (to_rgb(marked), "Pixel (100,100)"),
        (img_rgb, "BGR→RGB"),
        (to_rgb(cropped), "Cropped"),
        (to_rgb(resized), "Resized 400×300"),
        (to_rgb(flipped), "Flipped Horizontal"),
    ]
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    save_fig(base / "08_all_results.png")
    print("Lab 2 done ->", base)


# ---------------------------------------------------------------------------
# LAB 3 — Arithmetic ops + negative + white space + bounding boxes
# ---------------------------------------------------------------------------
def make_numpy_images(h: int, w: int) -> dict[str, np.ndarray]:
    """Synthetic images matching uploaded image size (BGR uint8)."""
    # Grayscale gradient as 3-channel
    grad = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
    grad_bgr = cv2.cvtColor(grad.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Checkerboard
    tile = 40
    yy, xx = np.indices((h, w))
    checker = (((yy // tile) + (xx // tile)) % 2) * 255
    checker_bgr = cv2.cvtColor(checker.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Synthetic RGB
    syn = np.zeros((h, w, 3), dtype=np.uint8)
    syn[:, :, 2] = np.linspace(0, 255, w, dtype=np.uint8)  # R
    syn[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]  # G
    syn[:, :, 0] = 128  # B
    return {"gradient": grad_bgr, "checkerboard": checker_bgr, "synthetic_rgb": syn}


def side_by_side(path: Path, left: np.ndarray, right: np.ndarray, t1: str, t2: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(to_rgb(left))
    plt.title(t1)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(to_rgb(right))
    plt.title(t2)
    plt.axis("off")
    save_fig(path)


def lab3(img_bgr: np.ndarray) -> None:
    # Work on a manageable size for arithmetic demos
    work = cv2.resize(img_bgr, (600, 400), interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]
    synth = make_numpy_images(h, w)
    np_img = synth["gradient"]  # primary NumPy image for ops

    # Save sources
    src = ensure_dir(OUT / "Lab 3" / "_sources")
    save_bgr(src / "nic_resized.jpg", work)
    for name, im in synth.items():
        save_bgr(src / f"{name}.jpg", im)

    # Task 1: Addition
    t1 = ensure_dir(OUT / "Lab 3" / "Task 1")
    notes = ["Task 1 — Addition: increasing constant brightens the image (clipped at 255).\n"]
    for const in (10, 30, 60):
        nic_add = cv2.add(work, np.full_like(work, const))
        np_add = cv2.add(np_img, np.full_like(np_img, const))
        side_by_side(
            t1 / f"nic_add_{const}.png",
            work,
            nic_add,
            "NIC original",
            f"NIC + {const}",
        )
        side_by_side(
            t1 / f"numpy_add_{const}.png",
            np_img,
            np_add,
            "NumPy gradient",
            f"Gradient + {const}",
        )
        save_bgr(t1 / f"nic_add_{const}.jpg", nic_add)
        save_bgr(t1 / f"numpy_add_{const}.jpg", np_add)
        notes.append(f"Constant {const}: mean NIC before={work.mean():.1f}, after={nic_add.mean():.1f}")
    write_info(t1 / "observation.txt", "\n".join(notes) + "\n")

    # Task 2: Subtraction
    t2 = ensure_dir(OUT / "Lab 3" / "Task 2")
    notes = [
        "Task 2 — Subtraction: dark regions hit 0 first; larger constants darken more.\n"
    ]
    for const in (10, 30, 60):
        nic_sub = cv2.subtract(work, np.full_like(work, const))
        np_sub = cv2.subtract(np_img, np.full_like(np_img, const))
        zero_pct = 100.0 * np.mean(nic_sub == 0)
        side_by_side(
            t2 / f"nic_sub_{const}.png",
            work,
            nic_sub,
            "NIC original",
            f"NIC - {const}",
        )
        side_by_side(
            t2 / f"numpy_sub_{const}.png",
            np_img,
            np_sub,
            "NumPy gradient",
            f"Gradient - {const}",
        )
        save_bgr(t2 / f"nic_sub_{const}.jpg", nic_sub)
        save_bgr(t2 / f"numpy_sub_{const}.jpg", np_sub)
        notes.append(f"Constant {const}: ~{zero_pct:.1f}% of NIC pixels became 0")
    write_info(t2 / "observation.txt", "\n".join(notes) + "\n")

    # Task 3: Multiplication
    t3 = ensure_dir(OUT / "Lab 3" / "Task 3")
    notes = [
        "Task 3 — Multiplication: factor <1 darkens; >1 brightens; values clip at 255 (saturation).\n"
    ]
    for factor in (0.5, 1.2, 2.0):
        nic_mul = np.clip(work.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        np_mul = np.clip(np_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        sat = 100.0 * np.mean(nic_mul == 255)
        side_by_side(
            t3 / f"nic_mul_{factor}.png",
            work,
            nic_mul,
            "NIC original",
            f"NIC × {factor}",
        )
        side_by_side(
            t3 / f"numpy_mul_{factor}.png",
            np_img,
            np_mul,
            "NumPy gradient",
            f"Gradient × {factor}",
        )
        save_bgr(t3 / f"nic_mul_{factor}.jpg", nic_mul)
        save_bgr(t3 / f"numpy_mul_{factor}.jpg", np_mul)
        notes.append(f"Factor {factor}: saturated pixels ≈ {sat:.1f}%")
    write_info(t3 / "observation.txt", "\n".join(notes) + "\n")

    # Task 4: Division
    t4 = ensure_dir(OUT / "Lab 3" / "Task 4")
    notes = [
        "Task 4 — Division by value >1 reduces intensity (image becomes darker).\n"
    ]
    for div in (2, 3, 5):
        nic_div = (work.astype(np.float32) / div).clip(0, 255).astype(np.uint8)
        np_div = (np_img.astype(np.float32) / div).clip(0, 255).astype(np.uint8)
        side_by_side(
            t4 / f"nic_div_{div}.png",
            work,
            nic_div,
            "NIC original",
            f"NIC / {div}",
        )
        side_by_side(
            t4 / f"numpy_div_{div}.png",
            np_img,
            np_div,
            "NumPy gradient",
            f"Gradient / {div}",
        )
        save_bgr(t4 / f"nic_div_{div}.jpg", nic_div)
        save_bgr(t4 / f"numpy_div_{div}.jpg", np_div)
        notes.append(f"Divisor {div}: mean intensity {work.mean():.1f} → {nic_div.mean():.1f}")
    write_info(t4 / "observation.txt", "\n".join(notes) + "\n")

    # Task 5: Image Negative
    t5 = ensure_dir(OUT / "Lab 3" / "Task 5")
    nic_neg = 255 - work
    np_neg = 255 - np_img
    side_by_side(t5 / "nic_negative.png", work, nic_neg, "NIC original", "NIC negative")
    side_by_side(
        t5 / "numpy_negative.png", np_img, np_neg, "NumPy gradient", "Gradient negative"
    )
    save_bgr(t5 / "nic_negative.jpg", nic_neg)
    save_bgr(t5 / "numpy_negative.jpg", np_neg)
    write_info(
        t5 / "observation.txt",
        "Negative g=255-f reverses bright and dark regions; bright NIC text becomes dark.\n",
    )

    # Task 5b / Count White Spaces (applied to cropped NIC region)
    t5b = ensure_dir(OUT / "Lab 3" / "Task 5 Count White Spaces")
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    # White-ish pixels (empty / light background on NIC)
    white_mask = gray >= 200
    white_pct = 100.0 * white_mask.mean()
    white_vis = work.copy()
    white_vis[white_mask] = (0, 255, 255)  # highlight white in yellow
    side_by_side(
        t5b / "white_space_analysis.png",
        work,
        white_vis,
        "Cropped NIC",
        "White pixels highlighted (thresh>=200)",
    )
    save_bgr(t5b / "white_mask.jpg", (white_mask.astype(np.uint8) * 255))
    write_info(
        t5b / "white_space_stats.txt",
        f"Total pixels: {gray.size}\n"
        f"White pixels (intensity >= 200): {int(white_mask.sum())}\n"
        f"White space percentage: {white_pct:.2f}%\n"
        f"Mean intensity: {gray.mean():.2f}\n"
        f"Median intensity: {np.median(gray):.2f}\n",
    )

    # Task 6: Draw Bounding Boxes on three clear patches
    t6 = ensure_dir(OUT / "Lab 3" / "Task 6")
    boxed = work.copy()
    # Three regions on the resized NIC (photo / text / logo-ish areas)
    boxes = [
        (int(w * 0.05), int(h * 0.15), int(w * 0.28), int(h * 0.75)),  # left photo area
        (int(w * 0.32), int(h * 0.20), int(w * 0.75), int(h * 0.55)),  # center text
        (int(w * 0.70), int(h * 0.10), int(w * 0.95), int(h * 0.40)),  # right emblem area
    ]
    colors = [(0, 255, 0), (255, 0, 0), (0, 165, 255)]
    for (x1, y1, x2, y2), col in zip(boxes, colors):
        cv2.rectangle(boxed, (x1, y1), (x2, y2), col, 2)
    save_bgr(t6 / "highlighted_forest_area.jpg", boxed)  # name as required by lab
    side_by_side(
        t6 / "bounding_boxes.png",
        work,
        boxed,
        "Original",
        "Three highlighted patches",
    )
    write_info(
        t6 / "boxes.txt",
        "Three bounding boxes drawn on distinctive NIC regions "
        "(photo area, text block, emblem area).\n"
        + "\n".join(f"Box {i+1}: {b}" for i, b in enumerate(boxes))
        + "\n",
    )
    print("Lab 3 done")


# ---------------------------------------------------------------------------
# LAB 4 — Two-image arithmetic + bitwise + scenario
# ---------------------------------------------------------------------------
def lab4(img_bgr: np.ndarray) -> None:
    # Two compatible images derived from NIC (before/after style variants)
    a = cv2.resize(img_bgr, (500, 320), interpolation=cv2.INTER_AREA)
    # "After" variant: slightly darkened + blurred to simulate change
    b = cv2.convertScaleAbs(a, alpha=0.85, beta=-15)
    b = cv2.GaussianBlur(b, (5, 5), 0)

    # Task 1: Loading and preprocessing
    t1 = ensure_dir(OUT / "Lab 4" / "Task 1")
    save_bgr(t1 / "image1_before.jpg", a)
    save_bgr(t1 / "image2_after.jpg", b)
    side_by_side(t1 / "inputs.png", a, b, "Image 1 (before)", "Image 2 (after)")
    write_info(
        t1 / "compatibility.txt",
        f"Image1 shape={a.shape}, dtype={a.dtype}\n"
        f"Image2 shape={b.shape}, dtype={b.dtype}\n"
        f"Compatible: {a.shape == b.shape and a.dtype == b.dtype}\n",
    )

    # Task 2: Addition
    t2 = ensure_dir(OUT / "Lab 4" / "Task 2")
    added = cv2.add(a, b)
    save_bgr(t2 / "cv2_add.jpg", added)
    side_by_side(t2 / "addition.png", a, added, "Image A", "cv2.add(A, B)")
    # Manual verify one pixel
    y, x = 50, 50
    manual = np.clip(a[y, x].astype(np.int16) + b[y, x].astype(np.int16), 0, 255)
    write_info(
        t2 / "pixel_verify.txt",
        f"Pixel ({y},{x}) A={a[y,x]} B={b[y,x]}\n"
        f"Manual clip-sum={manual}\ncv2.add result={added[y,x]}\n",
    )

    # Task 3: Weighted addition
    t3 = ensure_dir(OUT / "Lab 4" / "Task 3")
    weights = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (wa, wb) in zip(axes, weights):
        blend = cv2.addWeighted(a, wa, b, wb, 0)
        save_bgr(t3 / f"blend_{wa}_{wb}.jpg", blend)
        ax.imshow(to_rgb(blend))
        ax.set_title(f"α={wa}, β={wb}")
        ax.axis("off")
    save_fig(t3 / "weighted_blends.png")
    write_info(
        t3 / "observation.txt",
        "Higher α keeps more of Image1; higher β emphasizes Image2.\n"
        "(0.5,0.5)=equal blend; (0.7,0.3)=more before; (0.3,0.7)=more after.\n",
    )

    # Task 4: Subtraction & absdiff
    t4 = ensure_dir(OUT / "Lab 4" / "Task 4")
    a_minus_b = cv2.subtract(a, b)
    b_minus_a = cv2.subtract(b, a)
    absdiff = cv2.absdiff(a, b)
    save_bgr(t4 / "A_minus_B.jpg", a_minus_b)
    save_bgr(t4 / "B_minus_A.jpg", b_minus_a)
    save_bgr(t4 / "absdiff.jpg", absdiff)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, im, title in zip(
        axes,
        [a_minus_b, b_minus_a, absdiff],
        ["A − B", "B − A", "absdiff(A,B)"],
    ):
        ax.imshow(to_rgb(im))
        ax.set_title(title)
        ax.axis("off")
    save_fig(t4 / "subtraction_compare.png")
    write_info(
        t4 / "observation.txt",
        "cv2.subtract clips negatives to 0 so A-B and B-A differ.\n"
        "absdiff keeps magnitude of change — useful for change detection.\n",
    )

    # Task 5–7: Bitwise on masks
    h, w = a.shape[:2]
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask1, (w // 3, h // 2), min(h, w) // 4, 255, -1)
    cv2.rectangle(
        mask2,
        (w // 2 - 60, h // 2 - 50),
        (w // 2 + 100, h // 2 + 70),
        255,
        -1,
    )

    t5 = ensure_dir(OUT / "Lab 4" / "Task 5")
    bit_and = cv2.bitwise_and(mask1, mask2)
    save_bgr(t5 / "mask1.jpg", mask1)
    save_bgr(t5 / "mask2.jpg", mask2)
    save_bgr(t5 / "bitwise_and.jpg", bit_and)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, im, title in zip(
        axes, [mask1, mask2, bit_and], ["Mask 1 (circle)", "Mask 2 (rect)", "AND"]
    ):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    save_fig(t5 / "bitwise_and.png")
    write_info(
        t5 / "observation.txt",
        "AND keeps only the overlapping (intersection) region of both masks.\n",
    )

    t6 = ensure_dir(OUT / "Lab 4" / "Task 6")
    bit_or = cv2.bitwise_or(mask1, mask2)
    save_bgr(t6 / "bitwise_or.jpg", bit_or)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, im, title in zip(
        axes, [mask1, mask2, bit_or], ["Mask 1", "Mask 2", "OR"]
    ):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    save_fig(t6 / "bitwise_or.png")
    write_info(
        t6 / "observation.txt",
        "OR combines both selected regions (union of circle and rectangle).\n",
    )

    t7 = ensure_dir(OUT / "Lab 4" / "Task 7")
    bit_not = cv2.bitwise_not(mask1)
    save_bgr(t7 / "bitwise_not.jpg", bit_not)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(mask1, cmap="gray")
    axes[0].set_title("Original mask")
    axes[0].axis("off")
    axes[1].imshow(bit_not, cmap="gray")
    axes[1].set_title("NOT (complement)")
    axes[1].axis("off")
    save_fig(t7 / "bitwise_not.png")
    write_info(
        t7 / "observation.txt",
        "NOT inverts 0↔255, producing the complementary mask.\n",
    )

    # Scenario: Tumor / change detection adapted to NIC before/after pair
    scen = ensure_dir(OUT / "Lab 4" / "Scenario Change Detection")
    change = cv2.absdiff(a, b)
    change_gray = cv2.cvtColor(change, cv2.COLOR_BGR2GRAY)
    _, change_mask = cv2.threshold(change_gray, 25, 255, cv2.THRESH_BINARY)
    highlighted = a.copy()
    highlighted[change_mask > 0] = (0, 0, 255)
    blend_roi = cv2.addWeighted(a, 0.6, b, 0.4, 0)
    roi = cv2.bitwise_and(a, a, mask=change_mask)

    save_bgr(scen / "change_absdiff.jpg", change)
    save_bgr(scen / "change_mask.jpg", change_mask)
    save_bgr(scen / "highlighted_changes.jpg", highlighted)
    save_bgr(scen / "blended_compare.jpg", blend_roi)
    save_bgr(scen / "masked_roi.jpg", roi)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        (to_rgb(a), "Before"),
        (to_rgb(b), "After"),
        (to_rgb(change), "Abs Diff"),
        (change_mask, "Change Mask"),
        (to_rgb(highlighted), "Highlighted"),
        (to_rgb(blend_roi), "Blend"),
    ]
    for ax, (im, title) in zip(axes.ravel(), panels):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    save_fig(scen / "scenario_overview.png")
    write_info(
        scen / "observation.txt",
        "Using NIC as Image1 and a darkened/blurred variant as Image2.\n"
        "absdiff + threshold highlights changed regions; blending aids visual compare.\n"
        "(Lab MRI scenario adapted to available NIC images as requested.)\n",
    )
    print("Lab 4 done")


def main() -> None:
    print("Loading NIC:", NIC_PATH)
    img = load_nic()
    print(f"Loaded shape={img.shape}")

    # Fresh output root
    ensure_dir(OUT)
    lab1(img)
    lab2(img)
    lab3(img)
    lab4(img)

    # Summary tree
    print("\n=== Output tree ===")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(p.relative_to(ROOT))
    print("\nAll labs complete. Results in:", OUT)


if __name__ == "__main__":
    main()
