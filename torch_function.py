import torch
from triton_kernels import (block_sparse_conv_1d_bwd_dx_kernel, 
                            block_sparse_conv_1d_bwd_dvalues_kernel, 
                            block_sparse_conv_1d_fwd_kernel)


def block_sparse_conv_1d_forward(x, coo_block_coords, values, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    x:                (B, C, T) float32
    coo_block_coords: (N_NONZERO_BLOCKS, 2) int32
    values:           (N_NONZERO_BLOCKS, BLOCK_SIZE_M, BLOCK_SIZE_M, KERNEL_SIZE) float32
    """
    B, n_channels, n_time_steps = x.shape
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    KERNEL_SIZE = values.shape[-1]

    # Allocate output (same shape as input, [B, C, T])
    y = torch.zeros_like(x)

    grid = (
        (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        B
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_fwd_kernel[grid](
            x, coo_block_coords, values, y,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS,
            num_warps=4
        )
    return y

def block_sparse_conv_1d_backward(dy, x, coo_block_coords, values, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    Backward pass for the block-sparse 1D convolution.
    
    dy:               (B, C, T) gradient of the output
    x:                (B, C, T) original input
    coo_block_coords: (N_NONZERO_BLOCKS, 2) int32
    values:           (N_NONZERO_BLOCKS, BLOCK_SIZE_M, BLOCK_SIZE_M, KERNEL_SIZE) float32
    Returns:
       dx:      gradient with respect to x (shape: [B, C, T])
       dvalues: gradient with respect to the kernel values (same shape as values)
    """
    B, n_channels, n_time_steps = x.shape
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    KERNEL_SIZE = values.shape[-1]

    dx = torch.zeros_like(x)
    dvalues = torch.zeros_like(values)

    grid = (
        (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        B
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dx_kernel[grid](
            dy, dx, coo_block_coords, values,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS,
            num_warps=4
        )
    
        block_sparse_conv_1d_bwd_dvalues_kernel[grid](
            dy, x, dvalues, coo_block_coords,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS,
            num_warps=4
        )

    return dx, dvalues

class BlockSparseConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coo_block_coords, values,
                BLOCK_SIZE_M, BLOCK_SIZE_N):
        ctx.save_for_backward(x, coo_block_coords, values)
        ctx.BLOCK_SIZE_M = BLOCK_SIZE_M
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N

        y = block_sparse_conv_1d_forward(x, coo_block_coords, values,
                                         BLOCK_SIZE_M, BLOCK_SIZE_N)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, coo_block_coords, values = ctx.saved_tensors
        BLOCK_SIZE_M = ctx.BLOCK_SIZE_M
        BLOCK_SIZE_N = ctx.BLOCK_SIZE_N
        dx, dvalues = block_sparse_conv_1d_backward(dy, x, coo_block_coords, values,
                                                    BLOCK_SIZE_M, BLOCK_SIZE_N)
        return dx, None, dvalues, None, None

def block_sparse_conv_1d_autograd(x, coo_block_coords, values,
                                  BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    Convenience API: call the autograd Function.
    """
    return BlockSparseConv1dFunction.apply(x, coo_block_coords, values, BLOCK_SIZE_M, BLOCK_SIZE_N)
