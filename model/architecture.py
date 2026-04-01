import random, itertools
import numpy as np
import torch
import torch.nn as nn

from model.data_config import ModalityConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)


# ── CELL B: ANFIS with proper 4-param Trapezoidal MF ─────────────────────────
class ANFIS(nn.Module):
    """
    TSK ANFIS with:
    - Gaussian MF: (mu, sigma) per feature per set
    - 4-param Trapezoidal MF: (a, b, c, d) where [b,c] is the flat top
    - Learnable MF mix (alpha): sigmoid-gated blend of Gaussian and Trap
    - Learnable t-norm: blend of product and min t-norms
    """
    def __init__(self, n_features: int, n_fuzzy_sets: int,
                 use_learnable_t_norm: bool = True):
        super().__init__()
        self.n_features   = n_features
        self.n_fuzzy_sets = n_fuzzy_sets
        self.n_rules      = n_fuzzy_sets ** n_features

        if self.n_rules > 128:
            import warnings
            warnings.warn(
                f"ANFIS rule explosion: {n_fuzzy_sets}^{n_features}={self.n_rules} rules. "
                f"Consider reducing n_fuzzy_sets or n_selected_feats.",
                UserWarning
            )

        # ── Gaussian MF parameters ────────────────────────────────────────────
        # Spread centres uniformly across [-2, 2] (standardised feature range)
        centres = torch.linspace(-1.5, 1.5, n_fuzzy_sets).unsqueeze(0).repeat(n_features, 1)
        self.mu_gauss    = nn.Parameter(centres + 0.1 * torch.randn_like(centres))
        self.sigma_gauss = nn.Parameter(torch.full((n_features, n_fuzzy_sets), 0.8))

        # ── 4-param Trapezoidal MF: a < b ≤ c < d ────────────────────────────
        # Initialise so that [b, c] is a narrow flat top centred at each mu
        # a, d: outer feet; b, c: inner shoulders
        self.trap_a = nn.Parameter(centres - 1.2 + 0.05 * torch.randn_like(centres))
        self.trap_b = nn.Parameter(centres - 0.4 + 0.05 * torch.randn_like(centres))
        self.trap_c = nn.Parameter(centres + 0.4 + 0.05 * torch.randn_like(centres))
        self.trap_d = nn.Parameter(centres + 1.2 + 0.05 * torch.randn_like(centres))

        # ── MF blend ─────────────────────────────────────────────────────────
        self.mf_mix = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5 blend

        # ── t-norm blend ──────────────────────────────────────────────────────
        self.t_norm_weight = (nn.Parameter(torch.tensor(0.0))
                              if use_learnable_t_norm else None)

        # ── TSK consequents ───────────────────────────────────────────────────
        self.consequent = nn.Parameter(
            torch.randn(self.n_rules, n_features + 1) * 0.01
        )
        self.rule_drop = nn.Dropout(0.1)

        all_rules = list(itertools.product(range(n_fuzzy_sets), repeat=n_features))
        self.register_buffer("rule_indices",
                             torch.tensor(all_rules, dtype=torch.long))

    def _gaussian_mf(self, xe):
        """xe: (B, F, 1) → (B, F, S)"""
        return torch.exp(
            -((xe - self.mu_gauss) ** 2) /
             (2 * self.sigma_gauss.clamp(min=1e-6) ** 2)
        )

    def _trap_mf(self, xe):
        """
        4-param trapezoid: 1 on [b,c], linearly rises on [a,b], falls on [c,d], 0 outside.
        Enforced ordering: a < b ≤ c < d via soft sorting.
        xe: (B, F, 1) → (B, F, S)
        """
        # Soft-sort to maintain a ≤ b ≤ c ≤ d (avoids degenerate trapezoids)
        params = torch.stack([self.trap_a, self.trap_b,
                               self.trap_c, self.trap_d], dim=-1)  # (F, S, 4)
        params = params.sort(dim=-1).values                         # (F, S, 4)
        a, b, c, d = params[..., 0], params[..., 1], params[..., 2], params[..., 3]

        #xe_ = xe.squeeze(-1)   # (B, F, S) after broadcasting

        left_slope  = ((xe - a) / (b - a + 1e-6)).clamp(0.0, 1.0)
        flat_top    = (xe >= b).float() * (xe <= c).float()
        right_slope = ((d - xe) / (d - c + 1e-6)).clamp(0.0, 1.0)

        return torch.max(torch.min(left_slope, right_slope), flat_top)

    def _membership(self, x):
        xe    = x.unsqueeze(-1)          # (B, F, 1) — broadcasts over S
        gauss = self._gaussian_mf(xe)
        trap  = self._trap_mf(xe)
        alpha = torch.sigmoid(self.mf_mix)
        return alpha * gauss + (1 - alpha) * trap

    def forward(self, x):
        B   = x.size(0)
        mem = self._membership(x)                     # (B, F, S)
        gathered = torch.stack(
            [mem[:, i, :][:, self.rule_indices[:, i]]
             for i in range(self.n_features)], dim=1) # (B, F, n_rules)

        if self.t_norm_weight is not None:
            w       = torch.sigmoid(self.t_norm_weight)
            firing  = (w * gathered.prod(dim=1)
                       + (1 - w) * gathered.min(dim=1)[0])
        else:
            firing = gathered.prod(dim=1)

        if self.training:
            firing = self.rule_drop(firing)

        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)
        x_aug       = torch.cat([torch.ones(B, 1, device=x.device), x], dim=1)
        rule_out    = (x_aug.unsqueeze(1) * self.consequent.unsqueeze(0)).sum(dim=2)
        out         = (norm_firing * rule_out).sum(dim=1, keepdim=True)

        return {"output": out, "firing": norm_firing, "membership": mem}


# ── 6b. Flat MLP Encoder (all modalities — pre-extracted .npz vectors) ────────
class FlatEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.3):
        super().__init__()
        hidden = min(256, input_dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),          # ← replaces BatchNorm1d
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.LayerNorm(64),              # ← replaces BatchNorm1d
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, latent_dim),
        )
        self.skip = nn.Linear(input_dim, latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.skip(x)


def build_encoder(mcfg: ModalityConfig) -> nn.Module:
    """All modalities use FlatEncoder — raw image processing is offline only."""
    return FlatEncoder(mcfg.input_dim, mcfg.latent_dim)

# ── CELL C: ANFIS-Aware Fusion Gate ──────────────────────────────────────────
class AgnosticHybridFusion(nn.Module):
    """
    Fusion model with gate that observes both stream logits.

    Changes vs original:
    - Gate input: [clinical_logit, firing_entropy, visual_logit, masks] instead of
                  [raw_visual_concat, raw_clinical_features, masks]
    - Entropy confidence uses numerically stable clamped log
    - Temperature params are logged during evaluation
    """
    def __init__(
        self,
        modality_configs      : list,
        clinical_dim          : int,
        n_fuzzy_sets          : int  = 2,
        use_entropy_fallback  : bool = True,
    ):
        super().__init__()
        self.mod_names            = [m.name for m in modality_configs]
        self.use_entropy_fallback = use_entropy_fallback
        n_mods = len(modality_configs)

        self.encoders = nn.ModuleDict(
            {m.name: build_encoder(m) for m in modality_configs}
        )

        visual_dim = sum(m.latent_dim for m in modality_configs)

        self.visual_predictor = nn.Sequential(
            nn.Linear(visual_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

        self.anfis = ANFIS(clinical_dim, n_fuzzy_sets)

        self.clin_temp = nn.Parameter(torch.ones(1))
        self.vis_temp  = nn.Parameter(torch.ones(1))

        # ── ANFIS-aware gate ──────────────────────────────────────────────────
        # Inputs: clinical_logit(1) + firing_entropy(1) + visual_logit(1) + masks(n_mods)
        gate_in_dim = 1 + 1 + 1 + n_mods
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1),
        )

    @staticmethod
    def entropy_confidence(logit: torch.Tensor) -> torch.Tensor:
        """1 - H(p) using numerically stable clamped log."""
        p  = torch.sigmoid(logit).clamp(1e-7, 1 - 1e-7)
        H  = -(p * p.log() + (1 - p) * (1 - p).log())
        return 1.0 - H

    def forward(self, batch: dict) -> tuple:
        # 1. Encode modalities
        masks, feats = [], []
        for name in self.mod_names:
            m        = batch[f"mask_{name}"]
            raw_feat = self.encoders[name](batch[f"mod_{name}"])
            feat     = torch.where(
                m.unsqueeze(1).bool().expand_as(raw_feat),
                raw_feat,
                raw_feat.detach()
            )
            feats.append(feat * m.unsqueeze(1))
            masks.append(m)

        # 2. Stream logits
        visual_concat  = torch.cat(feats, dim=1)
        visual_logit   = self.visual_predictor(visual_concat)
        anfis_out      = self.anfis(batch['clinical'])
        clinical_logit = anfis_out['output']

        # 3. ANFIS firing entropy as uncertainty signal
        firing          = anfis_out['firing']                        # (B, n_rules)
        firing_entropy  = -(firing * (firing + 1e-8).log()).sum(dim=-1, keepdim=True)
        # Normalise entropy to [0,1] by max possible entropy
        max_entropy     = torch.log(torch.tensor(firing.size(1), dtype=torch.float,
                                                  device=firing.device))
        firing_entropy  = firing_entropy / (max_entropy + 1e-8)      # (B, 1)

        # 4. Gate input: stream logits + ANFIS uncertainty + modality masks
        masks_vec  = torch.stack(masks, dim=-1)                      # (B, n_mods)
        gate_input = torch.cat(
            [clinical_logit, firing_entropy, visual_logit, masks_vec], dim=1
        )
        weights      = self.gate(gate_input)                         # (B, 2)
        wclin, wvis  = weights[:, :1], weights[:, 1:]

        # 5. Hard override: no imaging → 100% clinical
        has_images  = (masks_vec.sum(-1, keepdim=True) > 0).float()
        wvis        = wvis  * has_images
        wclin       = wclin * has_images + (1.0 - has_images)
        total_w     = wvis + wclin + 1e-8
        wvis, wclin = wvis / total_w, wclin / total_w

        # 6. Entropy-based confidence refinement
        if self.use_entropy_fallback:
            conf_vis  = self.entropy_confidence(visual_logit) * has_images
            conf_clin = self.entropy_confidence(clinical_logit)
            conf_tot  = conf_vis + conf_clin + 1e-8
            wvis      = 0.5 * wvis  + 0.5 * conf_vis  / conf_tot
            wclin     = 0.5 * wclin + 0.5 * conf_clin / conf_tot
            total_w   = wvis + wclin + 1e-8
            wvis, wclin = wvis / total_w, wclin / total_w

        # 7. Temperature scaling + fusion
        clin_t      = self.clin_temp.abs().clamp(min=0.1)
        vis_t       = self.vis_temp.abs().clamp(min=0.1)
        final_logit = wclin * (clinical_logit / clin_t) + \
                      wvis  * (visual_logit   / vis_t)

        fused_weights = torch.cat([wclin, wvis], dim=1)
        return final_logit, fused_weights, anfis_out
