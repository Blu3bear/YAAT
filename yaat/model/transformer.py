"""OnsetTransformer architecture for YAAT.

Reimplementation of the TensorHero OnsetTransformer — a transformer model
that takes windowed spectrogram frames around detected onsets and predicts
note contour events (plurality + motion pairs).

Architecture:
    - Input: 7-frame spectrogram windows (512 mel × 7 frames) per onset,
      flattened to 3584 dims, projected through a shared dense layer to 512.
    - Encoder: Standard transformer encoder (2 layers, 8 heads).
    - Decoder: Autoregressive transformer decoder, outputs over a 25-token
      vocabulary (13 pluralities + 9 motions + 3 special tokens).
"""

import math

import torch
import torch.nn as nn


# Special token indices
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

# Vocabulary layout:
#   0 = <sos>, 1 = <eos>, 2 = <pad>
#   3–15 = note pluralities (13 types)
#   16–24 = motions -4 to +4
VOCAB_SIZE = 25
NUM_PLURALITIES = 13
NUM_MOTIONS = 9
PLURALITY_OFFSET = 3
MOTION_OFFSET = 16


class OnsetInputEmbedding(nn.Module):
    """Projects flattened onset spectrogram windows to the model dimension.

    Each onset is represented by a 7-frame window of the mel spectrogram.
    The 7 frames are flattened and projected through a shared dense layer
    followed by a sigmoid activation.
    """

    def __init__(self, n_mels: int = 512, n_frames: int = 7, d_model: int = 512):
        super().__init__()
        self.input_dim = n_mels * n_frames  # 3584
        self.linear = nn.Linear(self.input_dim, d_model)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Flattened onset windows, shape (batch, seq_len, n_mels * n_frames).

        Returns:
            Projected embeddings, shape (batch, seq_len, d_model).
        """
        return self.activation(self.linear(x))


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Input with positional embeddings added.
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.embedding(positions)


class OnsetTransformer(nn.Module):
    """Transformer model for note contour prediction from onset-windowed spectrograms.

    Takes windowed spectrogram frames centered on detected onsets and
    autoregressively predicts pairs of (note_plurality, motion) tokens.
    """

    def __init__(
        self,
        embedding_size: int = 512,
        trg_vocab_size: int = VOCAB_SIZE,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        forward_expansion: int = 2048,
        dropout: float = 0.1,
        max_len: int = 50,
        n_mels: int = 512,
        n_frames: int = 7,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.d_model = embedding_size
        self.trg_vocab_size = trg_vocab_size

        # Encoder input: project flattened onset windows
        self.src_embedding = OnsetInputEmbedding(n_mels, n_frames, embedding_size)
        self.src_pos = PositionalEncoding(embedding_size, max_len)

        # Decoder input: embed discrete output tokens
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_pos = PositionalEncoding(embedding_size, max_len * 2 + 2)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=forward_expansion,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)

    def make_trg_mask(self, trg_len: int) -> torch.Tensor:
        """Create a causal mask for the decoder.

        Args:
            trg_len: Length of the target sequence.

        Returns:
            Upper-triangular boolean mask, shape (trg_len, trg_len).
        """
        mask = torch.triu(
            torch.ones(trg_len, trg_len, device=self.device), diagonal=1
        ).bool()
        return mask

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (teacher forcing).

        Args:
            src: Encoder input, shape (batch, src_len, n_mels * n_frames).
            trg: Decoder input tokens, shape (batch, trg_len).

        Returns:
            Logits over vocabulary, shape (batch, trg_len, vocab_size).
        """
        # Encode
        src_emb = self.src_pos(self.src_embedding(src))

        # Decode
        trg_emb = self.trg_pos(self.trg_embedding(trg))
        trg_mask = self.make_trg_mask(trg.shape[1])

        out = self.transformer(
            src_emb,
            trg_emb,
            tgt_mask=trg_mask,
        )

        return self.fc_out(out)

    @torch.no_grad()
    def predict(
        self,
        src: torch.Tensor,
        max_len: int = 102,
    ) -> list[int]:
        """Autoregressive inference: generate tokens until <eos> or max_len.

        Args:
            src: Encoder input, shape (1, src_len, n_mels * n_frames).
            max_len: Maximum number of output tokens.

        Returns:
            List of predicted token indices (excluding <sos>).
        """
        self.eval()

        # Encode source
        src_emb = self.src_pos(self.src_embedding(src))

        # Start with <sos>
        trg_tokens = [SOS_TOKEN]

        for _ in range(max_len):
            trg_tensor = torch.tensor(
                [trg_tokens], dtype=torch.long, device=self.device
            )
            trg_emb = self.trg_pos(self.trg_embedding(trg_tensor))
            trg_mask = self.make_trg_mask(len(trg_tokens))

            out = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask)
            logits = self.fc_out(out)

            # Greedy: take argmax of last position
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            trg_tokens.append(next_token)

            if next_token == EOS_TOKEN:
                break

        # Return tokens after <sos>, excluding <eos>
        result = trg_tokens[1:]
        if result and result[-1] == EOS_TOKEN:
            result = result[:-1]

        return result
