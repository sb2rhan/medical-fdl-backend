import nibabel as nib
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm


# --- PART 1: TENSOR EXTRACTION (unchanged from original) ---

def load_oasis_mri(hdr_path, target_shape=(128, 256, 256)):
    """
    Loads an OASIS .hdr/.img pair and returns a PyTorch-ready 4D tensor.
    Format: (Channel, Depth, Height, Width)
    """
    img = nib.load(hdr_path)
    data = img.get_fdata()

    tensor = torch.from_numpy(data).float()
    tensor = tensor.squeeze()  # Remove trailing 1s -> (256, 256, 128)

    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)  # (128, 256, 256)

    # Min-Max normalization to [0, 1]
    t_min, t_max = tensor.min(), tensor.max()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)

    tensor = tensor.unsqueeze(0)  # (1, 128, 256, 256)
    return tensor


# --- PART 2: 3D CNN FEATURE EXTRACTOR ---

class MRI3DFeatureExtractor(nn.Module):
    """
    A 3D CNN that maps an MRI volume (1, D, H, W) to a flat 1024-dim feature vector.

    Architecture:
        Block 1: Conv3d(1  -> 32)  + BN + ReLU + MaxPool  -> /2 spatial
        Block 2: Conv3d(32 -> 64)  + BN + ReLU + MaxPool  -> /2 spatial
        Block 3: Conv3d(64 -> 128) + BN + ReLU + MaxPool  -> /2 spatial
        Block 4: Conv3d(128-> 256) + BN + ReLU + MaxPool  -> /2 spatial
        Global Average Pool (spatial dims -> 1x1x1)
        FC: 256 -> 1024
        L2-normalize output
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )

        self.encoder = nn.Sequential(
            conv_block(1,   32),   # (1,128,256,256) -> (32,64,128,128)
            conv_block(32,  64),   # -> (64,32,64,64)
            conv_block(64,  128),  # -> (128,16,32,32)
            conv_block(128, 256),  # -> (256,8,16,16)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # -> (256,1,1,1)

        self.fc = nn.Sequential(
            nn.Flatten(),          # -> (256,)
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 1, D, H, W)
        Returns:
            features: L2-normalized tensor of shape (B, 1024)
        """
        x = self.encoder(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)

        # L2 normalize so all feature vectors live on the unit sphere
        # (makes downstream cosine-similarity comparisons valid)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


# --- PART 3: DATASET DISCOVERY ---

def find_oasis_img_files(disc1_root: str) -> list[dict]:
    """
    Recursively walks disc1/ and finds every processed MPRAGE .img file.

    Expected OASIS layout:
        disc1/
          OAS1_XXXX_MR1/
            PROCESSED/MPRAGE/SUBJ_111/
              OAS1_XXXX_MR1_mpr_n4_anon_sbj_111.img
    """
    pattern = os.path.join(
        disc1_root,
        "OAS1_*",
        "PROCESSED", "MPRAGE", "SUBJ_111",
        "*.img"
    )
    paths = sorted(glob.glob(pattern, recursive=False))

    records = []
    for p in paths:
        # Normalize slashes first (handles Windows backslash paths)
        normalized = p.replace("\\", "/")
        parts = normalized.split("/")
        # Subject ID is the OAS1_XXXX_MR1 folder — find it explicitly
        subject_id = next((part for part in parts if part.startswith("OAS1_")), None)
        if subject_id is None:
            print(f"  ⚠ Could not extract subject ID from: {p}, skipping.")
            continue
        records.append({"subject_id": subject_id, "img_path": p})#returning the dict

    return records


# --- PART 4: BATCH FEATURE EXTRACTION ---

def extract_and_save_features(
    disc1_root: str = "./all",
    output_dir: str = "./features",
    device: str | None = None,
    batch_size: int = 1,          # MRI volumes are huge; keep at 1 unless you have large GPU RAM
    use_pretrained_weights: bool = False,
    weights_path: str | None = None,
):
    """
    Walks all of disc1/, extracts a 1024-dim feature vector per subject,
    and saves them to output_dir.

    Output per subject:
        features/<subject_id>.npy     – numpy array, shape (1024,)

    Also writes:
        features/manifest.json        – maps subject_id -> .npy path + metadata
    """
    # ── Device selection ────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = MRI3DFeatureExtractor().to(device)
    model.eval()

    if use_pretrained_weights and weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {weights_path}")
    else:
        print("Using randomly-initialized weights (no pre-training).")
        print("For meaningful embeddings, fine-tune the model on a classification / "
              "reconstruction task first, then re-run extraction.")

    # ── Output dir ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # ── Discover subjects ───────────────────────────────────────────────────
    records = find_oasis_img_files(disc1_root)
    if not records:
        raise FileNotFoundError(
            f"No OASIS .img files found under '{disc1_root}'. "
            "Check the path and that the OASIS folder structure is intact."
        )
    print(f"Found {len(records)} subject(s) in '{disc1_root}'")

    manifest = {}
    failed   = []

    # ── Extraction loop ──────────────────────────────────────────────────────
    for rec in tqdm(records, desc="Extracting features"):
        subject_id = rec["subject_id"]
        img_path   = rec["img_path"]
        out_path   = os.path.join(output_dir, f"{subject_id}.npz")

        # Skip already-processed subjects (resume-friendly)
        if os.path.exists(out_path):
            manifest[subject_id] = {
                "feature_path": out_path,
                "img_path": img_path,
                "status": "skipped (already exists)",
            }
            continue

        try:
            # Load + preprocess
            tensor = load_oasis_mri(img_path)           # (1, 128, 256, 256)
            original_shape = np.array(tensor.shape)     # store before adding batch dim
            tensor = tensor.unsqueeze(0).to(device)     # (1, 1, 128, 256, 256)

            # Forward pass (no gradient tracking needed)
            with torch.no_grad():
                features = model(tensor)                # (1, 1024)

            feature_vec = features.squeeze(0).cpu().numpy()  # (1024,)

            # Sanity check
            assert feature_vec.shape == (1024,), \
                f"Unexpected feature shape: {feature_vec.shape}"

            import datetime
            np.savez(
                out_path,
                features       = feature_vec,                          # (1024,)
                subject_id     = np.array(subject_id),                 # scalar string
                img_path       = np.array(img_path),                   # scalar string
                original_shape = original_shape,                       # (4,) e.g. [1,128,256,256]
                timestamp      = np.array(datetime.datetime.now().isoformat()),
            )

            manifest[subject_id] = {
                "feature_path": out_path,
                "img_path": img_path,
                "status": "ok",
            }

        except Exception as e:
            print(f"\n  ✗ Failed on {subject_id}: {e}")
            failed.append({"subject_id": subject_id, "error": str(e)})
            manifest[subject_id] = {
                "feature_path": None,
                "img_path": img_path,
                "status": f"error: {e}",
            }

    # ── Save manifest ────────────────────────────────────────────────────────
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    ok_count   = sum(1 for v in manifest.values() if v["status"] == "ok")
    skip_count = sum(1 for v in manifest.values() if "skipped" in v["status"])
    fail_count = len(failed)

    print(f"\n{'='*50}")
    print(f"  Extraction complete!")
    print(f"  ✓ Extracted : {ok_count}")
    print(f"  ⏭  Skipped  : {skip_count}")
    print(f"  ✗ Failed    : {fail_count}")
    print(f"  Manifest    : {manifest_path}")
    print(f"{'='*50}")

    return manifest


# --- PART 5: QUICK SANITY CHECK UTILITY ---

def verify_features(output_dir: str = "./features"):
    """Load and print stats for all saved .npz feature files."""
    npz_files = sorted(glob.glob(os.path.join(output_dir, "*.npz")))
    if not npz_files:
        print("No .npz files found.")
        return

    print(f"Verifying {len(npz_files)} feature file(s)…\n")
    all_vecs = []
    for p in npz_files:
        data    = np.load(p, allow_pickle=True)
        vec     = data["features"]          # (1024,)
        subj    = str(data["subject_id"])
        shape   = tuple(data["original_shape"])
        ts      = str(data["timestamp"])
        norm    = np.linalg.norm(vec)
        all_vecs.append(vec)
        print(f"  {subj:30s}  features={vec.shape}  norm={norm:.4f}  "
              f"orig_shape={shape}  extracted={ts}")

    mat = np.stack(all_vecs)
    print(f"\nAll features matrix: {mat.shape}  (subjects × 1024)")
    print(f"Global mean: {mat.mean():.4f}  std: {mat.std():.4f}")


# --- PART 6: EXECUTION ---

if __name__ == "__main__":

    # ── 1. Quick architecture / shape sanity check ────────────────────────
    print("── Shape check ──")
    dummy = torch.randn(1, 1, 128, 256, 256)
    model = MRI3DFeatureExtractor()
    with torch.no_grad():
        out = model(dummy)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")   # Expected: (1, 1024)
    print(f"L2 norm of output: {out.norm(dim=1).item():.4f}")  # Expected: ~1.0
    print()

    # ── 2. Full dataset extraction ────────────────────────────────────────
    manifest = extract_and_save_features(
        disc1_root  = "./all",       # ← root of your OASIS disc1 folder
        output_dir  = "./features",    # ← where .npy files will be saved
        device      = None,            # auto-detect GPU/CPU
    )

    # ── 3. Verify saved features ──────────────────────────────────────────
    verify_features("./features")
