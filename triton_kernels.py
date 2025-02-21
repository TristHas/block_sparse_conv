import triton
import triton.language as tl

@triton.jit
def custom_dot(a, b):
    """
    Custom dot product for matrices a and b.
    a: shape [M, K]
    b: shape [K, N]
    Returns:
      c: shape [M, N]
    """
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    c = tl.zeros((M, N), dtype=a.dtype)
    for k in range(K):
        c += a[:, k:k+1] * b[k:k+1, :]
    return c

@triton.jit
def maybe_dot(a, b, BLOCK_SIZE_M: tl.constexpr):
    """
    Chooses between custom_dot and tl.dot based on BLOCK_SIZE_M.
    """
    if BLOCK_SIZE_M < 16:
        return custom_dot(a, b)
    else:
        return tl.dot(a, b)

@triton.jit
def block_sparse_conv_1d_fwd_kernel(
    x_ptr,               # float32* x [B, C, T]
    coo_ptr,             # int32*   coo_block_coords [N_NONZERO_BLOCKS, 2]
    values_ptr,          # float32* values [N_NONZERO_BLOCKS, BLOCK_SIZE_M, BLOCK_SIZE_M, KERNEL_SIZE]
    y_ptr,               # float32* output y [B, C, T]
    B: tl.int32,
    n_channels: tl.int32,
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.constexpr
):
    """
    Forward kernel for causal block-sparse convolution.
    """
    m_block = tl.program_id(0)  # output channel block index
    n_block = tl.program_id(1)  # time block index
    b_idx   = tl.program_id(2)  # batch index

    # Compute output channel and time indices for this block
    m_range = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_range < n_channels

    n_range = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_range < n_time_steps

    # Offsets in flattened tensors
    x_batch_offset = b_idx * n_channels * n_time_steps
    y_batch_offset = b_idx * n_channels * n_time_steps

    # Accumulator for y block
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over all nonzero kernel blocks
    for nzb in range(N_NONZERO_BLOCKS):
        r_block = tl.load(coo_ptr + nzb * 2 + 0)  # output block index
        c_block = tl.load(coo_ptr + nzb * 2 + 1)  # input block index

        # Process only if the kernel's output block equals our current m_block
        if r_block == m_block:
            in_channel_start = c_block * BLOCK_SIZE_M
            in_channels = in_channel_start + tl.arange(0, BLOCK_SIZE_M)
            in_mask = in_channels < n_channels

            # Loop over kernel taps
            for k in range(KERNEL_SIZE):
                # Compute offset into the kernel block (for tap k)
                out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]
                in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]
                val_index = (
                    nzb * BLOCK_SIZE_M * BLOCK_SIZE_M * KERNEL_SIZE
                    + out_idx * (BLOCK_SIZE_M * KERNEL_SIZE)
                    + in_idx * KERNEL_SIZE
                    + k
                )
                kernel_vals = tl.load(
                    values_ptr + val_index,
                    mask=(m_mask[:, None] & in_mask[None, :]),
                    other=0.0
                )  # shape: [BLOCK_SIZE_M, BLOCK_SIZE_M]

                # Compute the corresponding time index in x for this kernel tap (causal)
                t_idx = n_range + k - (KERNEL_SIZE - 1)
                valid_time = (t_idx >= 0) & (t_idx < n_time_steps)

                # Load a block of x: shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
                x_index = x_batch_offset + in_channels[:, None] * n_time_steps + t_idx[None, :]
                x_vals = tl.load(
                    x_ptr + x_index,
                    mask=(in_mask[:, None] & n_mask[None, :] & valid_time[None, :]),
                    other=0.0
                )

                # Use maybe_dot to select the appropriate dot function.
                acc += maybe_dot(kernel_vals, x_vals, BLOCK_SIZE_M)
    # Store result into y
    y_index = y_batch_offset + m_range[:, None] * n_time_steps + n_range[None, :]
    tl.store(
        y_ptr + y_index,
        acc,
        mask=(m_mask[:, None] & n_mask[None, :])
    )

@triton.jit
def block_sparse_conv_1d_bwd_dx_kernel(
    dy_ptr,            # float32* dy [B, C, T]
    dx_ptr,            # float32* dx [B, C, T]
    coo_ptr,           # int32* coo_block_coords [N_NONZERO_BLOCKS, 2]
    values_ptr,        # float32* values [N_NONZERO_BLOCKS, BLOCK_SIZE_M, BLOCK_SIZE_M, KERNEL_SIZE]
    B: tl.int32,
    n_channels: tl.int32,
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.constexpr
):
    """
    Backward kernel to compute gradient with respect to input x.
    """
    m_block = tl.program_id(0)  # output channel block index
    n_block = tl.program_id(1)  # time block index (for dy)
    b_idx   = tl.program_id(2)  # batch index

    m_range = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_range < n_channels

    n_range = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_range < n_time_steps

    dy_batch_offset = b_idx * n_channels * n_time_steps
    dy_index = dy_batch_offset + m_range[:, None] * n_time_steps + n_range[None, :]
    dy_block = tl.load(dy_ptr + dy_index, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)

    # Loop over all nonzero kernel blocks; only process those where output block matches.
    for nzb in range(N_NONZERO_BLOCKS):
        r_block = tl.load(coo_ptr + nzb * 2 + 0)
        c_block = tl.load(coo_ptr + nzb * 2 + 1)
        if r_block == m_block:
            in_channel_start = c_block * BLOCK_SIZE_M
            in_channels = in_channel_start + tl.arange(0, BLOCK_SIZE_M)
            in_mask = in_channels < n_channels

            for k in range(KERNEL_SIZE):
                # For each kernel tap k, compute the corresponding time index in x.
                t_idx = n_range + k - (KERNEL_SIZE - 1)
                valid_time = (t_idx >= 0) & (t_idx < n_time_steps)

                # Load kernel tap (same layout as in forward)
                out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]
                in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]
                val_index = (
                    nzb * BLOCK_SIZE_M * BLOCK_SIZE_M * KERNEL_SIZE
                    + out_idx * (BLOCK_SIZE_M * KERNEL_SIZE)
                    + in_idx * KERNEL_SIZE
                    + k
                )
                kernel_vals = tl.load(
                    values_ptr + val_index,
                    mask=(m_mask[:, None] & in_mask[None, :]),
                    other=0.0
                )

                # Use maybe_dot on the transposed kernel for the gradient.
                grad_x = maybe_dot(tl.trans(kernel_vals), dy_block, BLOCK_SIZE_M)

                # Scatter-add the gradient into dx.
                x_batch_offset = b_idx * n_channels * n_time_steps
                x_index = x_batch_offset + in_channels[:, None] * n_time_steps + t_idx[None, :]
                tl.atomic_add(dx_ptr + x_index, grad_x, mask=(in_mask[:, None] & n_mask[None, :] & valid_time[None, :]))
                

@triton.jit
def block_sparse_conv_1d_bwd_dvalues_kernel(
    dy_ptr,            # float32* dy [B, C, T]
    x_ptr,             # float32* x  [B, C, T]
    dvalues_ptr,       # float32* dvalues [N_NONZERO_BLOCKS, BLOCK_SIZE_M, BLOCK_SIZE_M, KERNEL_SIZE]
    coo_ptr,           # int32*   coo_block_coords [N_NONZERO_BLOCKS, 2]
    B: tl.int32,
    n_channels: tl.int32,
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.constexpr
):
    """
    Backward kernel to compute gradient with respect to kernel values.
    """
    m_block = tl.program_id(0)  # output channel block index (for dy)
    n_block = tl.program_id(1)  # time block index (for dy and corresponding x slice)
    b_idx   = tl.program_id(2)  # batch index

    m_range = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_range < n_channels

    n_range = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_range < n_time_steps

    dy_batch_offset = b_idx * n_channels * n_time_steps
    x_batch_offset = b_idx * n_channels * n_time_steps

    dy_index = dy_batch_offset + m_range[:, None] * n_time_steps + n_range[None, :]
    dy_block = tl.load(dy_ptr + dy_index, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)

    # Loop over nonzero kernel blocks whose output block equals the current m_block.
    for nzb in range(N_NONZERO_BLOCKS):
        r_block = tl.load(coo_ptr + nzb * 2 + 0)
        c_block = tl.load(coo_ptr + nzb * 2 + 1)
        if r_block == m_block:
            in_channel_start = c_block * BLOCK_SIZE_M
            in_channels = in_channel_start + tl.arange(0, BLOCK_SIZE_M)
            in_mask = in_channels < n_channels

            for k in range(KERNEL_SIZE):
                # Compute time indices for x corresponding to this kernel tap.
                t_idx = n_range + k - (KERNEL_SIZE - 1)
                valid_time = (t_idx >= 0) & (t_idx < n_time_steps)

                x_index = x_batch_offset + in_channels[:, None] * n_time_steps + t_idx[None, :]
                x_block = tl.load(
                    x_ptr + x_index,
                    mask=(in_mask[:, None] & n_mask[None, :] & valid_time[None, :]),
                    other=0.0
                )

                # Compute outer product between dy_block and x_block and reduce over time dimension.
                prod = dy_block[:, None, :] * x_block[None, :, :]  # shape: [BLOCK_SIZE_M, BLOCK_SIZE_M, BLOCK_SIZE_N]
                sum_val = tl.sum(prod, axis=2)  # shape: [BLOCK_SIZE_M, BLOCK_SIZE_M]

                dvalues_index = (
                    nzb * BLOCK_SIZE_M * BLOCK_SIZE_M * KERNEL_SIZE
                    + tl.arange(0, BLOCK_SIZE_M)[:, None] * (BLOCK_SIZE_M * KERNEL_SIZE)
                    + tl.arange(0, BLOCK_SIZE_M)[None, :] * KERNEL_SIZE
                    + k
                )
                tl.atomic_add(dvalues_ptr + dvalues_index, sum_val, mask=(m_mask[:, None] & in_mask[None, :]))

