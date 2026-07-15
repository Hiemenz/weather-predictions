"""ConvLSTM-based radar nowcasting trained on accumulated MRMS frames.

Architecture: a shallow Encoder-ConvLSTM-Decoder that takes N consecutive
MRMS crops as input and predicts the next frame. Compared against optical-flow
and persistence in the evaluator once enough frames are accumulated.

The model is deliberately kept small — the point is to beat optical flow with
real learned motion, not to win a benchmark. A few weeks of 2-minute MRMS
frames gives ~10k samples; that's enough to train a lightweight model and see
whether it beats naive extrapolation.

Training requires `poetry install --with convlstm`. PyTorch was chosen over
TF/Keras because it has prebuilt ARM/MPS wheels and runs on both the Mac
(MPS acceleration) and, in inference-only mode, the Pi (CPU).

Dataset builder (build_dataset) is intentionally separated from the training
loop so it can be called independently to inspect the data.

Needs: `poetry install --with convlstm`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weather_predictions.config import MODELS_DIR, MRMS_DATA_DIR
from weather_predictions.mrms_processing import load_mrms_grid
from weather_predictions.radar_nowcast import NO_ECHO_DBZ, _DBZ_MAX, _DBZ_MIN

log = logging.getLogger(__name__)

CONVLSTM_MODEL_PATH = MODELS_DIR / "mrms_convlstm.pt"
GRID_DIR = MRMS_DATA_DIR / "grids"

# Number of consecutive frames fed as input to the model.
SEQ_LEN = 4
# Spatial resolution of the training patches (pixels). Crops are tiled into
# non-overlapping patches of this size for training variety.
PATCH_SIZE = 128


class NotEnoughFramesError(RuntimeError):
    pass


def _normalize(dbz: np.ndarray) -> np.ndarray:
    """Map dBZ to [0, 1] using the same range as the optical-flow normalizer."""
    filled = np.nan_to_num(dbz, nan=NO_ECHO_DBZ)
    return np.clip((filled - _DBZ_MIN) / (_DBZ_MAX - _DBZ_MIN), 0.0, 1.0).astype(np.float32)


def _denormalize(x: np.ndarray) -> np.ndarray:
    return (x * (_DBZ_MAX - _DBZ_MIN) + _DBZ_MIN).astype(np.float32)


def build_dataset(
    lat: float,
    lon: float,
    radius_km: float = 300.0,
    grid_dir: Path = GRID_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) training arrays from all stored MRMS grids.

    X shape: (N, SEQ_LEN, PATCH_SIZE, PATCH_SIZE, 1)
    y shape: (N, PATCH_SIZE, PATCH_SIZE, 1)

    Each sample is a SEQ_LEN-frame sequence of normalized reflectivity patches;
    the target is the immediately following frame.

    Raises NotEnoughFramesError when fewer than SEQ_LEN + 1 frames exist.
    """
    from weather_predictions.mrms_processing import crop_to_region

    files = sorted(grid_dir.glob("MRMS_CONUS_*.npz"))
    if len(files) < SEQ_LEN + 1:
        raise NotEnoughFramesError(
            f"Need at least {SEQ_LEN + 1} MRMS frames to build a training dataset "
            f"(found {len(files)}). Run `weather mrms-fetch` repeatedly or "
            "`weather mrms-backfill` to accumulate more."
        )

    log.info("loading %d MRMS frames for dataset build", len(files))
    crops = []
    for f in files:
        try:
            frame = load_mrms_grid(f)
            crop = crop_to_region(frame, lat, lon, radius_km)
            crops.append(_normalize(crop["reflectivity_dbz"]))
        except Exception as e:
            log.warning("skipped %s: %s", f.name, e)

    if len(crops) < SEQ_LEN + 1:
        raise NotEnoughFramesError(
            f"Only {len(crops)} loadable crops after filtering — need {SEQ_LEN + 1}."
        )

    # Tile each crop into non-overlapping patches.
    xs, ys = [], []
    h, w = crops[0].shape
    for row_off in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for col_off in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            for i in range(len(crops) - SEQ_LEN):
                seq = np.stack(
                    [
                        c[row_off : row_off + PATCH_SIZE, col_off : col_off + PATCH_SIZE]
                        for c in crops[i : i + SEQ_LEN]
                    ]
                )[..., np.newaxis]
                target = crops[i + SEQ_LEN][row_off : row_off + PATCH_SIZE, col_off : col_off + PATCH_SIZE][
                    ..., np.newaxis
                ]
                xs.append(seq)
                ys.append(target)

    return np.stack(xs), np.stack(ys)


@dataclass
class TrainResult:
    n_samples: int
    n_epochs: int
    final_loss: float
    model_path: str


def train(
    lat: float,
    lon: float,
    radius_km: float = 300.0,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    grid_dir: Path = GRID_DIR,
    model_path: Path = CONVLSTM_MODEL_PATH,
) -> TrainResult:
    """Train (or retrain from scratch) the ConvLSTM on stored MRMS crops.

    Needs `poetry install --with convlstm` (PyTorch). On a Mac with an M-series
    chip, uses MPS acceleration automatically. On the Pi, CPU only.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for ConvLSTM training. "
            "Install with: poetry install --with convlstm"
        )

    X, y = build_dataset(lat, lon, radius_km, grid_dir)
    log.info("dataset: %d samples, patch size %dx%d", len(X), PATCH_SIZE, PATCH_SIZE)

    # (N, SEQ_LEN, H, W, 1) → (N, SEQ_LEN, 1, H, W) for PyTorch conv layers.
    X_t = torch.from_numpy(X.transpose(0, 1, 4, 2, 3))
    y_t = torch.from_numpy(y.transpose(0, 3, 1, 2))

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    log.info("training on %s", device)

    model = _ConvLSTMNowcaster(in_channels=1, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        final_loss = epoch_loss / len(dataset)
        log.info("epoch %d/%d loss=%.5f", epoch + 1, epochs, final_loss)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    log.info("saved model to %s", model_path)
    return TrainResult(
        n_samples=len(X),
        n_epochs=epochs,
        final_loss=final_loss,
        model_path=str(model_path),
    )


def predict(
    prev_crop: dict[str, Any],
    curr_crop: dict[str, Any],
    model_path: Path = CONVLSTM_MODEL_PATH,
) -> np.ndarray:
    """Run the trained ConvLSTM on two frames; returns a dBZ forecast array.

    Only uses the last two frames for now (SEQ_LEN=4 is padded with the
    current frame repeated). Once more history is accumulated, callers can
    pass a proper sequence.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. Install with: poetry install --with convlstm")

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained ConvLSTM found at {model_path}. Run `weather convlstm-train` first."
        )

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = _ConvLSTMNowcaster(in_channels=1, hidden_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    prev_n = _normalize(prev_crop["reflectivity_dbz"])
    curr_n = _normalize(curr_crop["reflectivity_dbz"])
    # Pad the sequence by repeating the earliest frame.
    frames = [prev_n] * (SEQ_LEN - 2) + [prev_n, curr_n]
    h, w = curr_n.shape
    X = np.stack(frames)[np.newaxis, :, np.newaxis, :, :]  # (1, SEQ_LEN, 1, H, W)
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()  # (1, 1, H, W)
    return _denormalize(pred[0, 0])


class _ConvLSTMCell(object):
    pass  # defined below with torch import deferred


# ── Model definition ────────────────────────────────────────────────────────
# Defined at module level so pickle/torch.save can find it by name, but imports
# are guarded so loading this module never forces a torch import.

def _build_model_classes() -> None:
    """Register _ConvLSTMNowcaster into module globals once torch is available."""
    import torch
    import torch.nn as nn

    class ConvLSTMCell(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
            super().__init__()
            pad = kernel_size // 2
            self.hidden_channels = hidden_channels
            self.gates = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=pad)

        def forward(
            self, x: "torch.Tensor", state: tuple["torch.Tensor", "torch.Tensor"] | None
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            b, _, h, w = x.shape
            if state is None:
                hx = x.new_zeros(b, self.hidden_channels, h, w)
                cx = x.new_zeros(b, self.hidden_channels, h, w)
            else:
                hx, cx = state
            combined = torch.cat([x, hx], dim=1)
            i, f, g, o = self.gates(combined).chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            cy = f * cx + i * g
            hy = o * torch.tanh(cy)
            return hy, cy

    class ConvLSTMNowcaster(nn.Module):
        def __init__(self, in_channels: int = 1, hidden_channels: int = 32) -> None:
            super().__init__()
            self.encoder = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
            self.cell = ConvLSTMCell(hidden_channels, hidden_channels)
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, in_channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, C, H, W)
            b, t, c, h, w = x.shape
            state = None
            for step in range(t):
                feat = torch.relu(self.encoder(x[:, step]))
                state = self.cell(feat, state)
            return self.decoder(state[0])

    globals()["_ConvLSTMNowcaster"] = ConvLSTMNowcaster


class _LazyModel:
    """Proxy: builds the real class on first call so module import never needs torch."""
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _build_model_classes()
        return globals()["_ConvLSTMNowcaster"](*args, **kwargs)


_ConvLSTMNowcaster = _LazyModel()  # type: ignore[assignment]
