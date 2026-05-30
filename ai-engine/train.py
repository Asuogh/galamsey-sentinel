#!/usr/bin/env python3
"""
train.py -- Phase 5: AI Model Training
=======================================
Trains a lightweight Convolutional Neural Network (CNN) to classify
Sentinel-2 image patches into land-cover classes using PyTorch.

REPORT CONTEXT -- CNN Architecture Choice:
------------------------------------------------------------------------------
We use a custom lightweight CNN rather than a pretrained ImageNet model
(e.g. ResNet-50) for the following reasons:

  1. INPUT MISMATCH: Pretrained ImageNet models expect 3-channel RGB images.
     Our patches are 8-band multispectral GeoTIFFs. Transfer learning from
     RGB weights to 8-channel inputs requires non-trivial weight surgery on
     the first convolutional layer and yields poor initialisation quality.

  2. DATASET SCALE: With dozens to hundreds of patches (not millions of
     ImageNet images), a large pretrained model would massively overfit.
     A lightweight custom CNN with appropriate regularisation (Dropout,
     BatchNorm) is better matched to our dataset scale.

  3. INTERPRETABILITY: A simple, purpose-built architecture is easier to
     explain in a project report and to ablate (test by removing layers).

Architecture overview (GalamseyNet):
  Input: [B, 8, 128, 128]   (batch, 8 bands, 128x128 pixels)
  Conv Block 1: Conv2d(8->32,  3x3) + BN + ReLU + MaxPool -> [B, 32,  64, 64]
  Conv Block 2: Conv2d(32->64, 3x3) + BN + ReLU + MaxPool -> [B, 64,  32, 32]
  Conv Block 3: Conv2d(64->128,3x3) + BN + ReLU + MaxPool -> [B, 128, 16, 16]
  AdaptiveAvgPool -> [B, 128, 4, 4]
  Flatten        -> [B, 2048]
  FC(2048->256)  + ReLU + Dropout(0.5)
  FC(256->num_classes)
  Output: [B, num_classes] logits

REPORT CONTEXT -- Why CrossEntropyLoss + Adam?
------------------------------------------------------------------------------
  CrossEntropyLoss combines LogSoftmax and NLLLoss in a single numerically
  stable operation. It is the standard loss for multi-class classification
  and works correctly for binary (2-class) classification too -- no need to
  switch to BCELoss, which would require sigmoid output and manual one-hot
  encoding.

  Adam (Adaptive Moment Estimation) adjusts the learning rate per-parameter
  using first and second moment estimates of gradients. It converges faster
  than SGD on small datasets and is robust to hyperparameter choice --
  suitable for a pipeline where rapid iteration is more important than
  squeezing out the last 0.1% accuracy.

REPORT CONTEXT -- GeoTIFF Loading via torchvision.datasets.ImageFolder:
------------------------------------------------------------------------------
  ImageFolder expects standard image files (PNG, JPEG) by default. Our
  patches are 8-band GeoTIFF files which PIL cannot open natively. We
  override ImageFolder's default loader with a custom rasterio-based loader
  that:
    1. Reads all 8 bands as a float32 numpy array.
    2. Returns a numpy array that our custom transform pipeline processes.
  We then use a custom ToTensor transform (instead of torchvision's) that
  correctly handles the 8-band array without the 3-channel RGB assumption.

Usage:
  python train.py [--patches_dir PATH] [--epochs N] [--batch_size N]
                  [--lr FLOAT] [--num_workers N] [--no_cuda] [--synthetic]
------------------------------------------------------------------------------
"""

import os
import sys
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

# -- Optional dependency warnings ---------------------------------------------
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn(
        "rasterio is not installed. Install it with: pip install rasterio\n"
        "Falling back to synthetic data mode for pipeline testing.",
        stacklevel=2,
    )

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger() -> logging.Logger:
    """
    Configures a structured logger that writes to both stdout and a log file.

    IMPORTANT -- Windows encoding fix:
    The StreamHandler is explicitly set to use UTF-8 encoding via a
    StreamHandler pointed at a reconfigured stdout. However, the safest
    approach on Windows is to avoid non-ASCII characters in log messages
    entirely, which this script does by using only standard ASCII characters
    (hyphens, equals signs, letters, digits).
    """
    logger = logging.getLogger("galamsey.train")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler -- INFO and above.
    # Use errors="replace" so any stray non-ASCII character is replaced
    # with "?" rather than raising a UnicodeEncodeError on Windows terminals
    # that use cp1252 or another non-UTF-8 codepage.
    ch = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                    errors="replace", closefd=False)
    )
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler -- DEBUG and above, always UTF-8.
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger


log = setup_logger()

# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GalamseyNet -- CNN training for Sentinel-2 patch classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        default=str(
            Path(__file__).parent.parent / "pipeline" / "extraction" / "patches"
        ),
        help="Root directory of the ImageFolder-structured patch dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size. Automatically clamped to dataset size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam optimiser learning rate.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of dataset to reserve for validation (0.0 to 1.0).",
    )
    parser.add_argument(
        "--min_val_samples",
        type=int,
        default=2,
        help=(
            "Minimum samples required to activate a validation split. "
            "Validation is skipped entirely if the dataset is smaller than this."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. Use 0 for single-process loading "
            "(safest on Windows and in small-dataset testing mode)."
        ),
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA even if a GPU is available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val splits.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Force synthetic data mode (ignores patches_dir). "
            "Useful for testing the full pipeline without real GeoTIFFs."
        ),
    )
    return parser.parse_args()

# =============================================================================
# CUSTOM GEOTIFF LOADER AND TRANSFORMS
# =============================================================================

def geotiff_loader(path: str) -> np.ndarray:
    """
    Custom image loader for ImageFolder that reads multi-band GeoTIFF files
    using rasterio and returns a float32 numpy array of shape [H, W, C].

    REPORT CONTEXT -- Why rasterio instead of PIL?
    --------------------------------------------------------------------------
    PIL supports common image formats (PNG, JPEG, BMP) but does not support
    multi-band GeoTIFF files natively. rasterio is a geospatial raster I/O
    library built on GDAL that reads GeoTIFF files correctly, including:
      - Multi-band files (any number of bands)
      - Float32 / Float64 / Int16 data types
      - LZW / Deflate compressed GeoTIFFs exported from GEE

    The loader normalises each band independently to [0, 1] using a robust
    percentile stretch (2nd-98th percentile) rather than a fixed min/max,
    which handles the wide dynamic range variation between optical bands
    (B2-B8) and SAR bands (VV, VH) in our 8-band stack.
    --------------------------------------------------------------------------

    Args:
        path: Absolute path to a .tif file.

    Returns:
        np.ndarray of shape [H, W, num_bands], dtype float32, values in [0, 1].
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(
            "rasterio is required to load GeoTIFF files. "
            "Install it with: pip install rasterio"
        )

    try:
        with rasterio.open(path) as src:
            # rasterio reads in [C, H, W] order -- transpose to [H, W, C].
            data = src.read().astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to open GeoTIFF '{path}': {exc}") from exc

    if data.shape[0] == 0:
        raise RuntimeError(f"GeoTIFF '{path}' has 0 bands.")

    # [C, H, W] -> [H, W, C]
    data = np.transpose(data, (1, 2, 0))

    # Per-band robust normalisation to [0, 1].
    # 2nd-98th percentile clips extreme outliers (cloud edges, sensor noise).
    for c in range(data.shape[2]):
        band   = data[:, :, c]
        p2     = np.percentile(band, 2)
        p98    = np.percentile(band, 98)
        drange = p98 - p2

        if drange > 1e-8:
            data[:, :, c] = np.clip((band - p2) / drange, 0.0, 1.0)
        else:
            data[:, :, c] = 0.0

    return data  # [H, W, C], float32, in [0, 1]


class GeoTiffToTensor:
    """
    Converts a [H, W, C] float32 numpy array to a [C, H, W] FloatTensor.

    Replaces torchvision.transforms.ToTensor() which assumes a uint8 PIL
    Image in [H, W, 3] format and incorrectly scales float arrays by 1/255.
    """

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        import torchvision.transforms.functional as TF
        return TF.to_tensor(img).float()


class ResizeArray:
    """
    Resizes a [H, W, C] numpy array to [target_size, target_size, C] using
    nearest-neighbour interpolation. Compatible with our numpy-based pipeline
    before the GeoTiffToTensor conversion step.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        if h == self.size and w == self.size:
            return img

        row_idx = np.round(np.linspace(0, h - 1, self.size)).astype(int)
        col_idx = np.round(np.linspace(0, w - 1, self.size)).astype(int)
        resized = img[np.ix_(row_idx, col_idx, np.arange(c))]
        return resized.astype(np.float32)


class RandomHorizontalFlipArray:
    """
    Randomly flips a [H, W, C] numpy array horizontally with probability p.

    REPORT CONTEXT -- Why augmentation for satellite imagery?
    --------------------------------------------------------------------------
    Satellite imagery has no canonical orientation -- a Galamsey pit looks
    identical regardless of cardinal direction. Horizontal and vertical flips
    are label-preserving augmentations that effectively multiply the apparent
    training set size without introducing unrealistic transformations. This
    is especially valuable during early pipeline testing with only 2 patches.
    --------------------------------------------------------------------------
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return img[:, ::-1, :].copy()
        return img


class RandomVerticalFlipArray:
    """Randomly flips a [H, W, C] numpy array vertically with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return img[::-1, :, :].copy()
        return img


class NormalizeArray:
    """
    Normalises a [C, H, W] float tensor using per-channel mean and std.
    Applied after GeoTiffToTensor so the input is already a tensor.

    Uses 0.5 mean and 0.5 std for all bands as a neutral starting point,
    mapping [0, 1] pixel values to [-1, 1].

    REPORT CONTEXT -- Production normalisation:
    --------------------------------------------------------------------------
    For production use, compute dataset-specific statistics over the full
    training set and replace the 0.5 defaults with those values:
      mean_per_band = dataset_tensor.mean(dim=(0, 2, 3))
      std_per_band  = dataset_tensor.std(dim=(0, 2, 3))
    --------------------------------------------------------------------------
    """

    def __init__(self, num_bands: int = 8):
        self.mean = torch.tensor([0.5] * num_bands).view(-1, 1, 1)
        self.std  = torch.tensor([0.5] * num_bands).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class ComposeArray:
    """Applies a list of transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def build_transforms(image_size: int = 128, augment: bool = True) -> ComposeArray:
    """
    Builds the full transform pipeline.

    Training   : Resize -> RandomHFlip -> RandomVFlip -> ToTensor -> Normalise
    Validation : Resize -> ToTensor -> Normalise
    """
    steps = [ResizeArray(image_size)]

    if augment:
        steps += [
            RandomHorizontalFlipArray(p=0.5),
            RandomVerticalFlipArray(p=0.5),
        ]

    steps += [
        GeoTiffToTensor(),
        NormalizeArray(num_bands=8),
    ]

    return ComposeArray(steps)

# =============================================================================
# SYNTHETIC DATASET
# =============================================================================

class SyntheticPatchDataset(torch.utils.data.Dataset):
    """
    Generates random 8-band 128x128 tensors with assigned class labels.

    Used when rasterio is unavailable, --synthetic is passed, or the
    patches directory is empty. Allows the full training pipeline to be
    validated end-to-end without requiring GEE data.

    REPORT CONTEXT -- Why synthetic data matters for pipeline testing:
    --------------------------------------------------------------------------
    In demo or CI environments where the GEE pipeline has not yet run,
    synthetic data lets us verify that the CNN architecture, loss function,
    optimiser, and weight-saving logic are all correct before committing to
    a full data collection run.
    --------------------------------------------------------------------------
    """

    def __init__(self, num_samples: int = 20, num_classes: int = 2,
                 num_bands: int = 8, image_size: int = 128):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.classes     = [f"class_{i}" for i in range(num_classes)]

        torch.manual_seed(42)
        self.data   = torch.randn(num_samples, num_bands, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx].item()

# =============================================================================
# IMAGEFOLDER DATASET WRAPPER
# =============================================================================

class GeoTiffImageFolder(datasets.ImageFolder):
    """
    Extends torchvision.datasets.ImageFolder to load 8-band GeoTIFF files.

    Overrides:
      - loader       : uses rasterio-based geotiff_loader
      - is_valid_file: accepts .tif and .tiff extensions only

    Class labels are inferred from subdirectory names sorted alphabetically:
      class_0_forest   -> label 0
      class_1_galamsey -> label 1
      class_2_water    -> label 2
    This matches the CLASS_LABELS convention from the Node.js pipeline.
    """

    def __init__(self, root: str, transform=None):
        super().__init__(
            root          = root,
            transform     = transform,
            loader        = geotiff_loader,
            is_valid_file = lambda p: p.lower().endswith((".tif", ".tiff")),
        )

# =============================================================================
# CNN ARCHITECTURE -- GalamseyNet
# =============================================================================

class ConvBlock(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.

    REPORT CONTEXT -- BatchNorm in convolutional blocks:
    --------------------------------------------------------------------------
    Batch Normalisation (Ioffe & Szegedy, 2015) normalises activations across
    the batch dimension, which:
      1. Reduces internal covariate shift, enabling higher learning rates.
      2. Acts as mild regularisation, reducing overfitting on small datasets.
      3. Makes training more stable when batch sizes are small (as ours are).
    --------------------------------------------------------------------------
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = kernel_size,
                padding     = kernel_size // 2,  # same padding preserves H, W
                bias        = False,              # bias is redundant with BN
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GalamseyNet(nn.Module):
    """
    Lightweight CNN for multi-class Sentinel-2 patch classification.

    Designed to be:
      - Small enough to train on CPU or a modest GPU with fewer than 50 patches.
      - Deep enough to learn spectral-spatial features from 8-band imagery.
      - Robust via BatchNorm and Dropout to avoid overfitting on tiny datasets.

    Total trainable parameters: approximately 2.1 million.
    """

    def __init__(self, num_bands: int = 8, num_classes: int = 2):
        super().__init__()

        self.num_bands   = num_bands
        self.num_classes = num_classes

        # Convolutional feature extractor.
        # Input: [B, 8, 128, 128]
        self.features = nn.Sequential(
            ConvBlock(num_bands,  32),   # -> [B, 32,  64, 64]
            ConvBlock(32,         64),   # -> [B, 64,  32, 32]
            ConvBlock(64,        128),   # -> [B, 128, 16, 16]
        )

        # AdaptiveAvgPool makes the classifier robust to slight patch size
        # variation (e.g. 127 vs 129 pixels from the GEE buffer operation).
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # -> [B, 128, 4, 4]

        # Fully-connected classifier. Input: 128 * 4 * 4 = 2048.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# =============================================================================
# DATASET LOADING AND SPLITTING
# =============================================================================

def load_dataset(patches_dir: str, use_synthetic: bool) -> tuple:
    """
    Loads the patch dataset. Falls back to synthetic data if the directory
    is empty, missing, or use_synthetic is True.

    Returns:
        (dataset, class_names, using_synthetic)
    """
    patches_path = Path(patches_dir)

    if use_synthetic or not RASTERIO_AVAILABLE:
        reason = "--synthetic flag" if use_synthetic else "rasterio not installed"
        log.warning(f"Using SYNTHETIC data ({reason}).")
        dataset = SyntheticPatchDataset(num_samples=20, num_classes=2)
        return dataset, dataset.classes, True

    if not patches_path.exists():
        log.warning(
            f"Patches directory not found: {patches_path}. "
            f"Falling back to synthetic data."
        )
        dataset = SyntheticPatchDataset(num_samples=20, num_classes=2)
        return dataset, dataset.classes, True

    tif_files = (
        list(patches_path.rglob("*.tif")) +
        list(patches_path.rglob("*.tiff"))
    )

    if len(tif_files) == 0:
        log.warning(
            f"No .tif files found under {patches_path}. "
            f"Falling back to synthetic data for pipeline testing."
        )
        dataset = SyntheticPatchDataset(num_samples=20, num_classes=2)
        return dataset, dataset.classes, True

    log.info(f"Loading GeoTIFF dataset from: {patches_path}")
    log.info(f"Found {len(tif_files)} .tif file(s).")

    train_transform = build_transforms(image_size=128, augment=True)

    try:
        dataset = GeoTiffImageFolder(
            root      = str(patches_path),
            transform = train_transform,
        )
    except FileNotFoundError as exc:
        log.error(f"ImageFolder error: {exc}")
        log.warning("Falling back to synthetic data.")
        dataset = SyntheticPatchDataset(num_samples=20, num_classes=2)
        return dataset, dataset.classes, True

    return dataset, dataset.classes, False


def split_dataset(
    dataset,
    val_split:       float,
    min_val_samples: int,
    seed:            int,
    using_synthetic: bool,
) -> tuple:
    """
    Splits the dataset into training and validation subsets.

    REPORT CONTEXT -- Graceful tiny-dataset handling:
    --------------------------------------------------------------------------
    With only 2 GeoTIFF patches during early pipeline testing, a standard
    80/20 split produces 1 train sample and 1 val sample. A DataLoader with
    batch_size > 1 on a 1-sample set is technically valid but produces
    meaningless accuracy figures.

    Strategy:
      1. Compute integer val_count = int(n * val_split).
      2. If val_count < min_val_samples (default 2), skip validation
         entirely and train on the full dataset.
      3. Batch size is separately clamped to min(batch_size, train_size).

    This guarantees the script never crashes regardless of dataset size.
    --------------------------------------------------------------------------

    Returns:
        (train_dataset, val_dataset_or_None, val_enabled)
    """
    n         = len(dataset)
    val_count = int(n * val_split)

    if val_count < min_val_samples:
        log.warning(
            f"Dataset has {n} sample(s). Validation requires at least "
            f"{min_val_samples} samples (computed val_count={val_count}). "
            f"SKIPPING validation -- training on full dataset."
        )
        return dataset, None, False

    train_count = n - val_count
    generator   = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_count, val_count], generator=generator
    )

    # Wrap the val subset with a non-augmenting transform for real GeoTIFFs.
    if not using_synthetic and hasattr(dataset, "samples"):
        val_transform = build_transforms(image_size=128, augment=False)

        class ValSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset    = subset
                self.transform = transform

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                raw_idx   = self.subset.indices[idx]
                raw_path  = self.subset.dataset.samples[raw_idx][0]
                label     = self.subset.dataset.targets[raw_idx]
                raw_array = geotiff_loader(raw_path)
                return self.transform(raw_array), label

        val_dataset = ValSubset(val_dataset, val_transform)

    log.info(
        f"Dataset split: {train_count} train / {val_count} val "
        f"(seed={seed})"
    )

    return train_dataset, val_dataset, True

# =============================================================================
# TRAINING AND VALIDATION LOOPS
# =============================================================================

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
    phase:     str,
) -> tuple:
    """
    Runs one training or validation epoch.

    Args:
        model     : The CNN model.
        loader    : DataLoader for this phase.
        criterion : Loss function (CrossEntropyLoss).
        optimizer : Optimizer, or None during validation.
        device    : torch.device.
        phase     : "train" or "val".

    Returns:
        (mean_loss, accuracy_percent)
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size     = inputs.size(0)
            total_loss    += loss.item() * batch_size
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            log.debug(
                f"  [{phase}] Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Correct: {(preds == labels).sum().item()}/{batch_size}"
            )

    mean_loss = total_loss / max(total_samples, 1)
    accuracy  = 100.0 * total_correct / max(total_samples, 1)
    return mean_loss, accuracy


# =============================================================================
# MAIN TRAINING ORCHESTRATOR
# =============================================================================

def train(args: argparse.Namespace) -> None:
    """
    Sets up data, model, loss, optimiser, runs the training loop,
    and saves the final model weights.
    """
    # Use only ASCII characters in all log separators and labels.
    log.info("=" * 65)
    log.info("GALAMSEY SENTINEL -- PHASE 5: CNN TRAINING")
    log.info("=" * 65)
    log.info(f"Patches dir  : {args.patches_dir}")
    log.info(f"Epochs       : {args.epochs}")
    log.info(f"Batch size   : {args.batch_size}")
    log.info(f"Learning rate: {args.lr}")
    log.info(f"Val split    : {args.val_split}")
    log.info(f"Seed         : {args.seed}")

    # -- Reproducibility ------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- Device selection -----------------------------------------------------
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        reason = "--no_cuda flag" if args.no_cuda else "no CUDA GPU detected"
        log.info(f"Device       : CPU ({reason})")
    else:
        device = torch.device("cuda")
        log.info(f"Device       : CUDA ({torch.cuda.get_device_name(0)})")

    # -- Load dataset ---------------------------------------------------------
    dataset, class_names, using_synthetic = load_dataset(
        args.patches_dir,
        use_synthetic=args.synthetic,
    )

    log.info(f"Dataset type : {'SYNTHETIC' if using_synthetic else 'GeoTIFF'}")
    log.info(f"Total samples: {len(dataset)}")
    log.info(f"Classes      : {class_names}")

    if not using_synthetic and hasattr(dataset, "targets"):
        class_counts = {}
        for label in dataset.targets:
            name = class_names[label]
            class_counts[name] = class_counts.get(name, 0) + 1
        for cls, cnt in class_counts.items():
            log.info(f"  {cls}: {cnt} sample(s)")

        counts = list(class_counts.values())
        if len(counts) > 1 and max(counts) / max(min(counts), 1) > 5:
            log.warning(
                "Class imbalance detected (ratio > 5x). "
                "Consider a weighted sampler or class weights in the loss."
            )

    # -- Train/val split ------------------------------------------------------
    train_dataset, val_dataset, val_enabled = split_dataset(
        dataset,
        val_split       = args.val_split,
        min_val_samples = args.min_val_samples,
        seed            = args.seed,
        using_synthetic = using_synthetic,
    )

    # -- Clamp batch size to training set size --------------------------------
    effective_batch = min(args.batch_size, len(train_dataset))
    if effective_batch < args.batch_size:
        log.warning(
            f"Batch size clamped from {args.batch_size} -> {effective_batch} "
            f"(training set only has {len(train_dataset)} sample(s))."
        )

    # -- DataLoaders ----------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size  = effective_batch,
        shuffle     = True,
        num_workers = args.num_workers,
        drop_last   = False,
        pin_memory  = (device.type == "cuda"),
    )

    val_loader = None
    if val_enabled and val_dataset is not None:
        val_batch  = min(effective_batch, len(val_dataset))
        val_loader = DataLoader(
            val_dataset,
            batch_size  = val_batch,
            shuffle     = False,
            num_workers = args.num_workers,
            drop_last   = False,
            pin_memory  = (device.type == "cuda"),
        )

    log.info(f"Train batches: {len(train_loader)}")
    if val_loader:
        log.info(f"Val batches  : {len(val_loader)}")
    else:
        log.info("Val batches  : N/A (validation skipped)")

    # -- Model ----------------------------------------------------------------
    num_classes = len(class_names) if class_names else 2
    num_bands   = 8

    model = GalamseyNet(num_bands=num_bands, num_classes=num_classes).to(device)

    log.info("-" * 65)
    log.info("MODEL: GalamseyNet")
    log.info("-" * 65)
    log.info(f"  Input bands  : {num_bands}")
    log.info(f"  Num classes  : {num_classes}")
    log.info(f"  Parameters   : {model.count_parameters():,}")

    # -- Loss and optimiser ---------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = 1e-4,
    )

    # FIX 1: Removed `verbose=True` argument from ReduceLROnPlateau.
    # The `verbose` parameter was deprecated in PyTorch 2.2 and removed in
    # later releases. To monitor LR changes, we log the current LR manually
    # at the end of each epoch by reading optimizer.param_groups[0]["lr"].
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 2,
    )

    log.info("Loss         : CrossEntropyLoss")
    log.info(f"Optimiser    : Adam (lr={args.lr}, weight_decay=1e-4)")
    log.info("LR scheduler : ReduceLROnPlateau (patience=2, factor=0.5)")
    log.info("-" * 65)

    # -- Training loop --------------------------------------------------------
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc   = 0.0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # FIX 2: All separator lines use only ASCII hyphens, no Unicode.
        log.info(f"\n-- Epoch {epoch}/{args.epochs} " + "-" * 45)

        # Training phase.
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, phase="train"
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        log.info(
            f"  TRAIN | Loss: {train_loss:.4f} | Accuracy: {train_acc:.1f}%"
        )

        # Validation phase.
        if val_loader is not None:
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None, device, phase="val"
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            log.info(
                f"  VAL   | Loss: {val_loss:.4f} | Accuracy: {val_acc:.1f}%"
            )

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                log.info(f"  New best val accuracy: {best_val_acc:.1f}%")
        else:
            scheduler.step(train_loss)

        # Log current LR manually (replaces the removed verbose=True output).
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        log.info(f"  LR: {current_lr:.2e} | Epoch time: {epoch_time:.1f}s")

    total_time = time.time() - training_start
    log.info(f"\nTraining complete in {total_time:.1f}s")

    # -- Save model weights ---------------------------------------------------
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "galamsey_model.pth"

    checkpoint = {
        "model_state_dict" : model.state_dict(),
        "num_bands"        : num_bands,
        "num_classes"      : num_classes,
        "class_names"      : class_names,
        "epochs_trained"   : args.epochs,
        "final_train_loss" : history["train_loss"][-1],
        "final_train_acc"  : history["train_acc"][-1],
        "best_val_acc"     : best_val_acc if val_enabled else None,
        "using_synthetic"  : using_synthetic,
        "training_args"    : vars(args),
        "saved_at"         : datetime.now().isoformat(),
        "pytorch_version"  : torch.__version__,
    }

    torch.save(checkpoint, model_path)

    # -- Final summary --------------------------------------------------------
    log.info("=" * 65)
    log.info("TRAINING SUMMARY")
    log.info("=" * 65)
    log.info(f"  Model saved     : {model_path}")
    log.info(f"  Data type       : {'SYNTHETIC' if using_synthetic else 'Real GeoTIFF'}")
    log.info(f"  Total samples   : {len(dataset)}")
    log.info(f"  Classes         : {class_names}")
    log.info(f"  Epochs          : {args.epochs}")
    log.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    log.info(f"  Final train acc : {history['train_acc'][-1]:.1f}%")

    if val_enabled and history["val_acc"]:
        log.info(f"  Best val acc    : {best_val_acc:.1f}%")
    else:
        log.info("  Val accuracy    : N/A (dataset too small for split)")

    log.info(f"  Total time      : {total_time:.1f}s")
    log.info("-" * 65)
    log.info("  To load this model for inference:")
    log.info(f"    ckpt  = torch.load(r'{model_path}')")
    log.info(f"    model = GalamseyNet(num_bands={num_bands}, num_classes={num_classes})")
    log.info( "    model.load_state_dict(ckpt['model_state_dict'])")
    log.info( "    model.eval()")
    log.info("=" * 65)

    if using_synthetic:
        log.warning(
            "SYNTHETIC DATA MODE: weights are NOT trained on real satellite data. "
            "Re-run after populating pipeline/extraction/patches/ with real GeoTIFFs."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)