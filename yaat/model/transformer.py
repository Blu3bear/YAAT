"""TensorHero Transformer architecture for YAAT.

Reimplementation of the TensorHero Transformer (Model 1) that matches
the model13 checkpoint. Takes 4-second spectrogram segments as input
and autoregressively predicts interleaved (time, note) token pairs.

Architecture:
    - Input: Spectrogram frames (512-dim each), passed individually through
      a shared dense layer (Linear 512->512 + Sigmoid).
    - Encoder: Standard transformer encoder (2 layers, 8 heads, ff=2048).
    - Decoder: Autoregressive decoder over a 435-token vocabulary:
        Tokens 0-31:   Note values (0=no note, 1-31=chords/notes)
        Tokens 32-431: Time bins (10ms each within a 4s window)
        Token 432:     <sos>
        Token 433:     <eos>
        Token 434:     <pad>
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Special token indices (Model 1 / model13 encoding)
SOS_TOKEN = 432
EOS_TOKEN = 433
PAD_TOKEN = 434

# Vocabulary layout
VOCAB_SIZE = 435
NOTE_RANGE = range(0, 32)  # Tokens 0-31: note values
TIME_RANGE = range(32, 432)  # Tokens 32-431: time bins (0-399)
TIME_OFFSET = 32  # Subtract from time token to get bin index


class InputEmbedding(nn.Module):
    """Projects continuous spectrogram frames to the model dimension.

    Each 512-dim spectrogram frame is passed through a shared dense layer
    followed by sigmoid activation.  Matches the tensor-hero InputEmbedding.

    State dict key: ``src_spec_embedding.linear.0.*``
    """

    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on a single spectrogram frame (or batched frames).

        Args:
            x: Shape (..., embedding_size).

        Returns:
            Same shape, projected through dense + sigmoid.
        """
        return self.linear(x)


class TensorHeroTransformer(nn.Module):
    """Transformer model matching tensor-hero's Model 1 (model13 checkpoint).

    Takes 4-second spectrogram segments (512 mel bins x 400 time frames)
    and autoregressively decodes interleaved (time_token, note_token) pairs.

    State dict keys are named to exactly match the model13.pt checkpoint:
        src_spec_embedding, src_position_embedding, trg_word_embedding,
        trg_position_embedding, transformer, fc_out.
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
        max_len: int = 500,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.trg_vocab_size = trg_vocab_size

        # Source: continuous spectrogram -> dense + sigmoid (per frame)
        self.src_spec_embedding = InputEmbedding(embedding_size)

        # Positional embeddings (learnable)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        # Target: discrete token vocabulary
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)

        # Core transformer
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
        self.dropout = nn.Dropout(dropout)

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create source padding mask.

        Marks positions beyond 400 frames as padding (True = ignore).

        Args:
            src: Source spectrogram, shape (batch, n_mels, src_len).

        Returns:
            Boolean mask, shape (batch, src_len).
        """
        src_len = src.shape[2]
        mask = torch.zeros(
            src.shape[0], min(400, src_len), dtype=torch.bool, device=self.device
        )
        if src_len > 400:
            pad_mask = torch.ones(
                src.shape[0], src_len - 400, dtype=torch.bool, device=self.device
            )
            mask = torch.cat((mask, pad_mask), dim=1)
        return mask

    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        """Embed source spectrogram frames.

        Passes each spectrogram frame (512-dim) through the shared
        InputEmbedding and adds positional embeddings.

        Args:
            src: Shape (batch, n_mels, src_len).

        Returns:
            Embedded source, shape (batch, src_len, embedding_size).
        """
        batch, n_mels, src_len = src.shape

        # Process each frame through InputEmbedding
        out_list = []
        for t in range(src_len):
            frame = src[:, :, t]  # (batch, 512)
            embedded = self.src_spec_embedding(frame)  # (batch, 512)
            out_list.append(embedded.unsqueeze(2))  # (batch, 512, 1)

        # Concatenate: (batch, 512, src_len)
        src_embed = torch.cat(out_list, dim=2)

        # Add positional embedding
        positions = torch.arange(src_len, device=self.device)
        positions = positions.unsqueeze(1).expand(src_len, batch)  # (src_len, batch)
        pos_embed = self.src_position_embedding(positions)  # (src_len, batch, 512)

        # src_embed is (batch, 512, src_len), pos_embed is (src_len, batch, 512)
        embedded = self.dropout(src_embed + pos_embed.permute(1, 2, 0))

        # Transpose to (batch, src_len, 512) for the transformer
        return embedded.permute(0, 2, 1)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (teacher forcing).

        Args:
            src: Spectrogram, shape (batch, n_mels, src_len).
            trg: Target tokens, shape (batch, trg_len).

        Returns:
            Logits over vocabulary, shape (batch, trg_len, vocab_size).
        """
        batch = src.shape[0]
        trg_len = trg.shape[1]

        # Source embedding + positional
        src_padding_mask = self._make_src_mask(src)
        embed_src = self._embed_src(src)

        # Target embedding + positional
        trg_positions = (
            torch.arange(trg_len, device=self.device)
            .unsqueeze(1)
            .expand(trg_len, batch)
            .permute(1, 0)
        )
        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        # Causal mask
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_len).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )

        return self.fc_out(out)

    @torch.no_grad()
    def predict(
        self,
        src: torch.Tensor,
        max_len: int = 500,
    ) -> list[int]:
        """Autoregressive inference: generate tokens until <eos> or max_len.

        Args:
            src: Spectrogram, shape (1, n_mels, src_len).
            max_len: Maximum number of output tokens.

        Returns:
            List of predicted token indices (excluding <sos> and <eos>).
        """
        self.eval()

        # Pre-compute source embedding
        embed_src = self._embed_src(src)
        src_padding_mask = self._make_src_mask(src)

        # Start with <sos>
        trg_tokens = [SOS_TOKEN]

        for _ in range(max_len):
            trg_tensor = torch.tensor(
                [trg_tokens], dtype=torch.long, device=self.device
            )
            trg_len = len(trg_tokens)

            trg_positions = torch.arange(trg_len, device=self.device).unsqueeze(0)
            embed_trg = self.dropout(
                self.trg_word_embedding(trg_tensor)
                + self.trg_position_embedding(trg_positions)
            )

            trg_mask = self.transformer.generate_square_subsequent_mask(trg_len).to(
                self.device
            )

            out = self.transformer(
                embed_src,
                embed_trg,
                src_key_padding_mask=src_padding_mask,
                tgt_mask=trg_mask,
            )
            logits = self.fc_out(out)

            next_token = int(torch.argmax(logits[0, -1, :]).item())

            if next_token == EOS_TOKEN:
                break

            trg_tokens.append(next_token)

        # Return tokens after <sos>
        result = trg_tokens[1:]
        return result
