"""
Multimodal TTM Fusion Model.

Architecture:
  1. Video stream  : I3D-R50 → 512-d embedding per clip   (frozen pretrained)
  2. Audio stream  : ResNet-18 → 512-d embedding per clip  (frozen pretrained)
  3. Projection    : each stream projected to embed_dim=256
  4. Concatenation : [video_proj; audio_proj] → 512-d per time step
  5. BiLSTM        : hidden=256 (×2 dirs = 512 out), layers=2, dropout=0.3
  6. Attention     : Bahdanau additive over all time steps → context vector
  7. Classifier    : Linear(512, 2) → TTM logits

Input (per utterance, T clips):
  video_feats : [B, T, 512]   — I3D embeddings per clip
  audio_feats : [B, T, 512]   — ResNet-18 embeddings per clip

Output:
  logits  : [B, 2]   (for classification)
  attn_w  : [B, T]   (attention weights, for interpretability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttentionBlock(nn.Module):
    """
    Single Bahdanau attention block with residual-style refinement.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [B, T, hidden_dim]
        Returns:
            context: [B, hidden_dim]
            weights: [B, T]
        """
        energy = self.v(torch.tanh(self.Wh(h)))            # [B, T, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)      # [B, T]
        context = (weights.unsqueeze(-1) * h).sum(dim=1)    # [B, hidden_dim]
        context = self.drop(context)
        return context, weights


class BahdanauAttention(nn.Module):
    """
    Additive attention over a sequence of LSTM hidden states.

    Supports stacking multiple Bahdanau blocks while preserving the
    final output shape of context and weights.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0, debug: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.debug = debug

        if num_layers == 1:
            self.layer = BahdanauAttentionBlock(hidden_dim, dropout=dropout)
        else:
            self.layers = nn.ModuleList([
                BahdanauAttentionBlock(hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ])

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [B, T, hidden_dim]
        Returns:
            context: [B, hidden_dim]
            weights: [B, T]
        """
        if self.num_layers == 1:
            if self.debug:
                print(f"Using {self.num_layers} Bahdanau attention layer")
            return self.layer(h)

        if self.debug:
            print(f"Using {self.num_layers} stacked Bahdanau attention layers")

        x = h
        skip = x
        final_context = None
        final_weights = None

        for i, layer in enumerate(self.layers):
            context, weights = layer(x)
            final_context, final_weights = context, weights
            context_expanded = context.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, hidden_dim]
            x = x + context_expanded

            if (i + 1) % 2 == 0:
                x = x + skip
                skip = x

        return final_context, final_weights


# NEW CODE: Cross-Modal Attention Fusion Module
class CrossModalAttentionBlock(nn.Module):
    """
    Single cross-modal attention block with residual connections and feedforward.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,           # 8→4: halves attention params, better for small data
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),   # 4×→2×: cuts FFN params in half
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, v: torch.Tensor, a: torch.Tensor):
        """
        Args:
            v: [B, T, embed_dim]
            a: [B, T, embed_dim]
        Returns:
            v: [B, T, embed_dim]
        """
        attn_out, _ = self.attn(v, a, a)
        v = v + attn_out
        v = self.norm1(v)

        ff_out = self.ff(v)
        v = v + ff_out
        v = self.norm2(v)
        return v


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing video and audio features.

    Supports stacking multiple cross-attention blocks with residual and skip
    connections, while preserving the original output shape.
    """

    def __init__(self, embed_dim: int, num_layers: int = 1, dropout: float = 0.0, debug: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.debug = debug
        # Always use CrossModalAttentionBlock so every layer has norm + ff + residual,
        # even in the single-layer case (raw MHA alone lacks these stabilisers).
        self.layers = nn.ModuleList([
            CrossModalAttentionBlock(embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def _attend(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Apply stacked cross-attention layers with skip connections."""
        x    = query
        skip = x
        for i, layer in enumerate(self.layers):
            x = layer(x, key_value)
            if (i + 1) % 2 == 0:
                x = x + skip
                skip = x
        return x

    def forward(self, v: torch.Tensor, a: torch.Tensor):
        """
        Bidirectional cross-modal attention with shared weights.

        Video queries attend to audio KV *and* audio queries attend to video KV
        using the exact same layer parameters.  Weight sharing means zero extra
        params while giving both streams cross-modal context.  Replaces the
        previous asymmetric design (cat[modified_v, original_a]) with
        cat[modified_v, modified_a], both informed by the other modality.

        Args:
            v: [B, T, embed_dim] video features
            a: [B, T, embed_dim] audio features

        Returns:
            fused: [B, T, embed_dim*2]  — same shape as before
        """
        if self.debug:
            print(f"Using {self.num_layers} bidirectional cross-modal attention layer(s)")

        v_mod = self._attend(v, a)   # video queries attend to audio KV
        a_mod = self._attend(a, v)   # audio queries attend to video KV (shared weights)

        fused = torch.cat([v_mod, a_mod], dim=-1)   # [B, T, embed_dim*2]
        return fused


class TTMFusionModel(nn.Module):
    """
    Bidirectional LSTM + Bahdanau attention over fused video + audio features.

    Args:
        video_dim  : I3D embedding size (512)
        audio_dim  : ResNet-18 embedding size (512)
        proj_dim   : projection dim per stream before concat (256)
        lstm_hidden: BiLSTM hidden units per direction (256 → 512 total)
        lstm_layers: stacked BiLSTM layers (2)
        num_classes: output classes (2)
        dropout    : applied inside BiLSTM and before classifier
        fusion_type: fusion method - "concat" (default) or "cross_attn"
        use_no_audio_token: learn a dedicated embedding for missing-audio clips
    """

    def __init__(
        self,
        video_dim:   int = 512,
        audio_dim:   int = 512,
        proj_dim:    int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_classes: int = 2,
        dropout:     float = 0.3,
        fusion_type: str = "concat",  # NEW: fusion type parameter
        cross_attn_layers: int = 1,
        bahdanau_layers: int = 1,
        use_no_audio_token: bool = True,
        debug:       bool = False,
    ):
        super().__init__()

        self.fusion_type = fusion_type  # NEW: store fusion type
        self.debug = debug
        self.audio_dim = audio_dim

        # Learned fallback embedding for clips with no audio.
        # When audio_feats is all-zero (missing .npy), replacing with a
        # trainable token lets the model learn a consistent representation
        # for the "no audio" case rather than attending to zero vectors —
        # which corrupts cross-modal attention by producing uniform scores.
        if use_no_audio_token:
            self.no_audio_token = nn.Parameter(torch.zeros(audio_dim))
        else:
            self.no_audio_token = None

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = proj_dim * 2   # 512

        # NEW: instantiate cross-attention only when needed to preserve
        # backward compatibility for default concat-only checkpoint loading.
        self.cross_attn = None
        if fusion_type == "cross_attn":
            self.cross_attn = CrossModalAttention(
                proj_dim,
                num_layers=cross_attn_layers,
                dropout=dropout,
                debug=debug,
            )

        self.bilstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        bilstm_out = lstm_hidden * 2   # 512

        # Normalise LSTM outputs before attention: prevents high-variance
        # hidden states from dominating attention scores on small datasets.
        self.lstm_norm = nn.LayerNorm(bilstm_out)

        self.attention  = BahdanauAttention(
            bilstm_out,
            num_layers=bahdanau_layers,
            dropout=dropout,
            debug=debug,
        )
        self.drop       = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(bilstm_out, num_classes)

    def forward(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
    ):
        """
        Args:
            video_feats: [B, T, 512]
            audio_feats: [B, T, 512]  — zero tensor when audio embedding is absent
        Returns:
            logits:  [B, 2]
            attn_w:  [B, T]
        """
        # Replace all-zero audio windows with the learned no-audio token so that
        # the cross-modal attention path receives a meaningful query/key rather
        # than attending uniformly to a zero sequence.
        if self.no_audio_token is not None:
            # [B]: True when the entire audio window is zeros (missing .npy file)
            audio_missing = (audio_feats.abs().sum(dim=(1, 2)) == 0)
            if audio_missing.any():
                token = self.no_audio_token.view(1, 1, self.audio_dim)  # [1,1,D]
                token = token.expand(audio_feats.size(0), audio_feats.size(1), -1)
                mask  = audio_missing[:, None, None].expand_as(audio_feats)
                audio_feats = torch.where(mask, token, audio_feats)

        v = self.video_proj(video_feats)   # [B, T, 256]
        a = self.audio_proj(audio_feats)   # [B, T, 256]
        
        # MODIFIED CODE: Conditional fusion based on fusion_type
        if self.fusion_type == "concat":
            # Original concatenation-based fusion
            x = torch.cat([v, a], dim=-1)      # [B, T, 512]
        elif self.fusion_type == "cross_attn":
            # New cross-modal attention fusion
            x = self.cross_attn(v, a)          # [B, T, 512]
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        # Debug print if requested
        if self.debug:
            print(f"Using fusion: {self.fusion_type}")

        h, _ = self.bilstm(x)              # [B, T, 512]
        h    = self.lstm_norm(h)           # stabilise before attention

        context, attn_w = self.attention(h)  # [B, 512], [B, T]
        logits = self.classifier(self.drop(context))  # [B, 2]

        return logits, attn_w

    def count_parameters(self) -> dict:
        total    = sum(p.numel() for p in self.parameters())
        trained  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trained}


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test both fusion types
    for fusion_type in ["concat", "cross_attn"]:
        print(f"\n--- Testing fusion_type: {fusion_type} ---")
        model  = TTMFusionModel(fusion_type=fusion_type).to(device)
        B, T   = 4, 16

        v = torch.randn(B, T, 512, device=device)
        a = torch.randn(B, T, 512, device=device)

        logits, attn_w = model(v, a)
        print(f"video_feats : {v.shape}")
        print(f"audio_feats : {a.shape}")
        print(f"logits      : {logits.shape}")
        print(f"attn_weights: {attn_w.shape}")
        p = model.count_parameters()
        print(f"Params — total: {p['total']/1e6:.2f}M  trainable: {p['trainable']/1e6:.2f}M")
