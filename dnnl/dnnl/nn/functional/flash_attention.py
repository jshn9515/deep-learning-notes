import math
from numbers import Real

import torch
from torch import Tensor

__all__ = [
    'flash_attention_v1_forward',
    'flash_attention_v1_backward',
]


def _validate_block_size(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f'`{name}` must be an integer.')
    if value <= 0:
        raise AssertionError(f'`{name}` must be greater than 0.')


def _validate_dropout(dropout: float) -> None:
    if not isinstance(dropout, Real):
        raise AssertionError('`dropout` must be a real number.')
    if not 0.0 <= dropout < 1.0:
        raise AssertionError('`dropout` must satisfy 0 <= dropout < 1.')


def _validate_scale(scale: float | None) -> None:
    if scale is None:
        return
    if not isinstance(scale, Real):
        raise AssertionError('`scale` must be a real number or None.')
    if not math.isfinite(scale) or scale <= 0.0:
        raise AssertionError('`scale` must be a finite positive number.')


def _validate_query_key_value(query: Tensor, key: Tensor, value: Tensor) -> None:
    if not isinstance(query, Tensor):
        raise AssertionError('`query` must be a torch.Tensor.')
    if not isinstance(key, Tensor):
        raise AssertionError('`key` must be a torch.Tensor.')
    if not isinstance(value, Tensor):
        raise AssertionError('`value` must be a torch.Tensor.')

    if query.ndim != key.ndim or query.ndim != value.ndim:
        raise AssertionError(
            '`query`, `key`, and `value` must have the same number of dimensions.'
        )
    if query.ndim not in (2, 3):
        raise AssertionError('`query`, `key`, and `value` must be 2D or 3D tensors.')
    if (
        not query.is_floating_point()
        or not key.is_floating_point()
        or not value.is_floating_point()
    ):
        raise AssertionError(
            '`query`, `key`, and `value` must be floating-point tensors.'
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise AssertionError('`query`, `key`, and `value` must have the same dtype.')
    if key.device != query.device or value.device != query.device:
        raise AssertionError('`query`, `key`, and `value` must be on the same device.')
    if key.shape != query.shape or value.shape != query.shape:
        raise AssertionError('`query`, `key`, and `value` must have the same shape.')
    if query.size(-2) == 0:
        raise AssertionError('sequence length must be greater than 0.')
    if query.size(-1) == 0:
        raise AssertionError('embedding dimension must be greater than 0.')


def _validate_gradient(query: Tensor, gradient: Tensor) -> None:
    if not isinstance(gradient, Tensor):
        raise AssertionError('`gradient` must be a torch.Tensor.')
    if gradient.shape != query.shape:
        raise AssertionError('`gradient` must have the same shape as `query`.')
    if gradient.dtype != query.dtype:
        raise AssertionError('`gradient` must have the same dtype as `query`.')
    if gradient.device != query.device:
        raise AssertionError('`gradient` must be on the same device as `query`.')


def flash_attention_v1_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    Br: int,
    Bc: int,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    dropout: float = 0.0,
    upcast: bool = True,
) -> Tensor:
    """
    Compute Flash Attention v1 forward pass with IO-aware block-wise computation.

    Implements an efficient attention mechanism that reduces HBM (high-bandwidth
    memory) accesses by materializing attention matrices block-wise in SRAM,
    following the algorithm from "Flash Attention: Fast and Memory-Efficient
    Exact Attention with IO-Awareness" (Dao et al., 2022).

    Args:
        query (Tensor): Query tensor of shape (N, d) or (B, N, d).
        key (Tensor): Key tensor of shape (N, d) or (B, N, d).
        value (Tensor): Value tensor of shape (N, d) or (B, N, d).
        Br (int): Block size for row dimension (queries).
        Bc (int): Block size for column dimension (keys).
        is_causal (bool, default: False): If True, applies causal masking to prevent
            attending to future tokens. Default is False.
        scale (float, optional): Scaling factor for attention scores. If None,
            defaults to 1/sqrt(d). Default is None.
        dropout (float, default: 0.0): Dropout probability applied to attention
            weights. Default is 0.0.
        upcast (bool, default: True): If True, upcasts float16/bfloat16 inputs to
            float32 for numerical stability. Default is True.

    Returns:
        Attention output tensor with the same leading dimensions as query.

    Raises:
        AssertionError: If query, key, value have mismatched dimensions or incorrect shapes.
    """
    _validate_query_key_value(query, key, value)
    _validate_block_size('Br', Br)
    _validate_block_size('Bc', Bc)
    _validate_dropout(dropout)
    _validate_scale(scale)

    original_dtype = query.dtype

    squeeze_batch = query.ndim == 2
    if squeeze_batch:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

    B, N, d = query.size()

    dtype = (
        torch.float32
        if (upcast and query.dtype in (torch.float16, torch.bfloat16))
        else query.dtype
    )
    Q = query.to(dtype=dtype)
    K = key.to(dtype=dtype)
    V = value.to(dtype=dtype)

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # line 3: initialize O, l, m in HBM
    O = torch.zeros(B, N, d, dtype=dtype, device=Q.device)
    l = torch.zeros(B, N, dtype=dtype, device=Q.device)
    m = torch.full((B, N), -math.inf, dtype=dtype, device=Q.device)

    # Split counts
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    # line 6: for j in [1..Tc]
    for j in range(Tc):
        k_start = j * Bc
        k_end = min((j + 1) * Bc, N)

        # line 7: read K_j, V_j from HBM
        Kj = K[:, k_start:k_end]
        Vj = V[:, k_start:k_end]

        # line 8: for i in [1..Tr]
        for i in range(Tr):
            q_start = i * Br
            q_end = min((i + 1) * Br, N)

            # line 9: read Q_i, O_i, l_i, m_i from HBM
            Qi = Q[:, q_start:q_end]
            Oi = O[:, q_start:q_end]
            li = l[:, q_start:q_end]
            mi = m[:, q_start:q_end]

            # line 10: S_ij = tau * Q_i K_j^T
            Sij = (Qi @ Kj.transpose(-2, -1)) * scale

            # Optional causal mask: positions in Qi correspond to global indices
            # [q_start..q_end), keys correspond to [k_start..k_end).
            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                # line 11: mask where key position > query position
                Sij = Sij.masked_fill(k_idx > q_idx, -math.inf)

            # line 12: m̃_ij = rowmax(S_ij), P̃_ij = exp(S_ij - m̃_ij),
            # l̃_ij = rowsum(P̃_ij)
            mij_tilde = Sij.max(dim=-1).values
            Pij_tilde = torch.where(
                torch.isfinite(mij_tilde).unsqueeze(-1),
                (Sij - mij_tilde.unsqueeze(-1)).exp(),
                torch.zeros_like(Sij),
            )
            lij_tilde = Pij_tilde.sum(dim=-1)

            # line 13: m_new = max(m_i, m̃_ij)
            mi_new = mi.maximum(mij_tilde)

            # l_new = exp(m_i - m_new)*l_i + exp(m̃_ij - m_new)*l̃_ij
            alpha = (mi - mi_new).exp()
            beta = (mij_tilde - mi_new).exp()
            li_new = alpha * li + beta * lij_tilde

            if dropout > 0.0:
                # line 14: (optional) dropout on P̃_ij before using it in O_i update
                mask = torch.rand_like(Pij_tilde) > dropout
                Pij_tilde = Pij_tilde * mask / (1.0 - dropout)

            # line 15:
            # O_i <- diag(l_new)^(-1) * ( diag(l_i)*exp(m_i-m_new)*O_i
            #         + exp(m̃_ij-m_new) * P̃_ij V_j )
            # Note: diag(...) is just per-row scaling -> broadcasting.
            Oi_new = (
                (alpha.unsqueeze(-1) * li.unsqueeze(-1) * Oi)
                + beta.unsqueeze(-1) * (Pij_tilde @ Vj)
            ) / li_new.unsqueeze(-1)

            # line 16: write O_i, l_i, m_i back to HBM
            O[:, q_start:q_end] = Oi_new
            l[:, q_start:q_end] = li_new
            m[:, q_start:q_end] = mi_new

    # Cast back to input dtype
    output = O.to(dtype=original_dtype)

    if squeeze_batch:
        return output.squeeze(0)

    return output


def flash_attention_v1_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gradient: Tensor,
    Br: int,
    Bc: int,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    dropout: float = 0.0,
    upcast: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Backward pass for Flash Attention v1.

    Computes gradients with respect to query, key, and value given the gradient
    of the output. Uses block-wise computation to reduce HBM (High Bandwidth Memory)
    accesses.

    Args:
        query (Tensor): Query tensor of shape (N, d) or (B, N, d).
        key (Tensor): Key tensor of shape (N, d) or (B, N, d).
        value (Tensor): Value tensor of shape (N, d) or (B, N, d).
        gradient (Tensor): Gradient of output tensor with the same shape as query.
        Br (int): Row block size for query blocks.
        Bc (int): Column block size for key/value blocks.
        is_causal (bool, default: False): If True, applies causal mask.
            Default: False.
        scale (float, optional): Scaling factor for attention scores (typically
            1/sqrt(d)). If None, computed as 1/sqrt(d). Default: None.
        dropout (float, default: 0.0): Dropout probability applied during forward
            pass. If > 0, backward is not supported. Default: 0.0.
        upcast (bool, default: True): If True, upcasts float16/bfloat16 inputs to
            float32 for numerics. Default: True.

    Returns:
        Tuple of (dQ, dK, dV), each with the same shape as query, key, and value.

    Raises:
        NotImplementedError: If dropout > 0.0 (exact backward with dropout mask needed).
        AssertionError: If tensor shapes are incompatible.

    Algorithm:
        1. Forward pass: Recompute attention scores, softmax, and outputs in blocks.
        2. Compute delta term D_i = sum(dO_i * O_i) for each query position.
        3. Backward pass: For each (Q_i, K_j) block pair:
           - Recover normalized attention probabilities P_ij from row-wise stats
             (m_i, l_i)
           - Compute dV += P_ij^T @ dO_i
           - Compute dP_ij = dO_i @ V_j^T
           - Compute dS_ij = P_ij * (dP_ij - D_i) (softmax gradient)
           - Accumulate dQ, dK gradients scaled by attention scores
    """
    _validate_query_key_value(query, key, value)
    _validate_gradient(query, gradient)
    _validate_block_size('Br', Br)
    _validate_block_size('Bc', Bc)
    _validate_dropout(dropout)
    _validate_scale(scale)

    original_dtype = query.dtype

    squeeze_batch = query.ndim == 2
    if squeeze_batch:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        gradient = gradient.unsqueeze(0)

    B, N, d = query.size()

    if dropout > 0.0:
        raise NotImplementedError(
            'Exact backward with dropout needs the same forward dropout mask to be saved.'
        )

    dtype = (
        torch.float32
        if (upcast and query.dtype in (torch.float16, torch.bfloat16))
        else query.dtype
    )
    Q = query.to(dtype=dtype)
    K = key.to(dtype=dtype)
    V = value.to(dtype=dtype)
    dO = gradient.to(dtype=dtype)

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    O = torch.zeros(B, N, d, dtype=dtype, device=Q.device)
    l = torch.zeros(B, N, dtype=dtype, device=Q.device)
    m = torch.full((B, N), -math.inf, dtype=dtype, device=Q.device)

    for j in range(Tc):
        k_start = j * Bc
        k_end = min((j + 1) * Bc, N)

        Kj = K[:, k_start:k_end]
        Vj = V[:, k_start:k_end]

        for i in range(Tr):
            q_start = i * Br
            q_end = min((i + 1) * Br, N)

            Qi = Q[:, q_start:q_end]
            Oi = O[:, q_start:q_end]
            li = l[:, q_start:q_end]
            mi = m[:, q_start:q_end]

            Sij = (Qi @ Kj.transpose(-2, -1)) * scale

            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                Sij = Sij.masked_fill(k_idx > q_idx, -math.inf)

            mij_tilde = Sij.max(dim=-1).values
            Pij_tilde = torch.where(
                torch.isfinite(mij_tilde).unsqueeze(-1),
                torch.exp(Sij - mij_tilde.unsqueeze(-1)),
                torch.zeros_like(Sij),
            )
            lij_tilde = Pij_tilde.sum(dim=-1)

            mi_new = torch.maximum(mi, mij_tilde)
            alpha = torch.exp(mi - mi_new)
            beta = torch.exp(mij_tilde - mi_new)
            li_new = alpha * li + beta * lij_tilde

            Oi_new = (
                alpha.unsqueeze(-1) * li.unsqueeze(-1) * Oi
                + beta.unsqueeze(-1) * (Pij_tilde @ Vj)
            ) / li_new.unsqueeze(-1)

            O[:, q_start:q_end] = Oi_new
            l[:, q_start:q_end] = li_new
            m[:, q_start:q_end] = mi_new

    # ------------------------------------------------------------
    # line 1 in many FA derivations:
    # D_i = dO_i · O_i   (row-wise inner product)
    # This is the delta term in softmax backward.
    # ------------------------------------------------------------
    D = (dO * O).sum(dim=-1)  # [B, N]

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # Same loop order as forward: outer j, inner i
    for j in range(Tc):
        k_start = j * Bc
        k_end = min((j + 1) * Bc, N)

        Kj = K[:, k_start:k_end]
        Vj = V[:, k_start:k_end]

        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)

        for i in range(Tr):
            q_start = i * Br
            q_end = min((i + 1) * Br, N)

            Qi = Q[:, q_start:q_end]
            dOi = dO[:, q_start:q_end]
            li = l[:, q_start:q_end]
            mi = m[:, q_start:q_end]
            Di = D[:, q_start:q_end]

            # Recompute S_ij
            Sij = (Qi @ Kj.transpose(-2, -1)) * scale

            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                Sij = Sij.masked_fill(k_idx > q_idx, -math.inf)

            # Recover normalized P_ij from global row stats m_i, l_i
            Pij = torch.exp(Sij - mi.unsqueeze(-1)) / li.unsqueeze(-1)

            # dV_j += P_ij^T dO_i
            dVj = dVj + Pij.transpose(-2, -1) @ dOi

            # dP_ij = dO_i V_j^T
            dPij = dOi @ Vj.transpose(-2, -1)

            # dS_ij = P_ij * (dP_ij - D_i)
            dSij = Pij * (dPij - Di.unsqueeze(-1))

            # masked positions should contribute zero gradient
            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                valid = k_idx <= q_idx
                dSij = dSij * valid

            # dQ_i += scale * dS_ij K_j
            dQ[:, q_start:q_end] += scale * (dSij @ Kj)

            # dK_j += scale * dS_ij^T Q_i
            dKj = dKj + scale * (dSij.transpose(-2, -1) @ Qi)

        dK[:, k_start:k_end] += dKj
        dV[:, k_start:k_end] += dVj

    dQ = dQ.to(dtype=original_dtype)
    dK = dK.to(dtype=original_dtype)
    dV = dV.to(dtype=original_dtype)

    if squeeze_batch:
        return dQ.squeeze(0), dK.squeeze(0), dV.squeeze(0)

    return dQ, dK, dV
