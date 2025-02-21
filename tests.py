from imports import *
sys.path.append("..")
from jphydro.routing.block_sparse_conv import BlockSparseCausalConv
from torch_function import block_sparse_conv_1d_autograd
from gen_data import generate_inputs

def test_fwd_triton_torchref(n_trees = 1,
                             max_heights = 3,
                             n_time_steps = 365,
                             max_delay = 10,
                             block_size = 32,
                             batch_size = 1,
                             BLOCK_SIZE_N = 64,
                             device = "cuda:0"):
    """
    """
    BLOCK_SIZE_M = block_size
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size)
    x = x.to(device)
    k = k.to(device)

    layer = BlockSparseCausalConv(k).to(device)
    out_ref = layer(x)
    out_new = block_sparse_conv_1d_autograd(x, k.block_indices, k.block_values, BLOCK_SIZE_M, BLOCK_SIZE_N)

    return torch.allclose(out_ref, out_new, atol=.001, rtol=.01)
    
def test_bwdx_triton_torchref(n_trees = 1,
                              max_heights = 3,
                              n_time_steps = 365,
                              max_delay = 10,
                              block_size = 32,
                              batch_size = 1,
                              BLOCK_SIZE_N = 64,
                              device = "cuda:0"):
    """
    """
    BLOCK_SIZE_M = block_size
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size, device)
    x = x.to(device).requires_grad_()
    k = k.to(device)

    out_ref = BlockSparseCausalConv(k).to(device)(x)
    out_new = block_sparse_conv_1d_autograd(x, k.block_indices, k.block_values, BLOCK_SIZE_M, BLOCK_SIZE_N)

    msk = torch.randn_like(out_ref)
    
    loss = ((out_ref*msk)**2).mean()
    loss.backward()
    dx1 = x.grad.clone()
    x.grad.zero_();

    loss = ((out_new*msk)**2).mean()
    loss.backward()
    dx2 = x.grad.clone()
    x.grad.zero_();
    
    return torch.allclose(dx1, dx2, atol=.001, rtol=.01)

def test_bwdw_triton_torchref(n_trees = 1,
                              max_heights = 3,
                              n_time_steps = 365,
                              max_delay = 10,
                              block_size = 32,
                              batch_size = 1,
                              BLOCK_SIZE_N = 64,
                              device = "cuda:0"):
    """
    """
    BLOCK_SIZE_M = block_size
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size, device)
    x = x.to(device)
    k = k.to(device)
    k.block_values.requires_grad_()
    
    out_ref = BlockSparseCausalConv(k).to(device)(x)
    out_new = block_sparse_conv_1d_autograd(x, k.block_indices, k.block_values, BLOCK_SIZE_M, BLOCK_SIZE_N)

    msk = torch.randn_like(out_ref)
    
    loss = ((out_ref*msk)**2).mean()
    loss.backward()
    dx1 = k.block_values.grad.clone()
    k.block_values.grad.zero_();

    loss = ((out_new*msk)**2).mean()
    loss.backward()
    dx2 = k.block_values.grad.clone()
    k.block_values.grad.zero_();
    
    return torch.allclose(dx1, dx2, atol=.001, rtol=.01)

def test_alls(
        N_TREES     = [1, 5],
        MAX_HEIGHTS = [3, 5],
        MAX_DELAYS  = [2, 10],
        BLOCK_SIZE  = [16, 32],
        DEVICES     = ["cuda:0", "cuda:5"],
        BATCH_SIZES = [1, 8],
        TIME_STEPS  = [14, 365]
    ):
    BLOCK_SIZE_N = 64
    res = []
    for n_time_steps in TIME_STEPS:
        for n_trees in N_TREES:
            for max_heights in MAX_HEIGHTS:
                for max_delay in MAX_DELAYS:
                    for block_size in BLOCK_SIZE:
                        for batch_size in BATCH_SIZES:
                            for device in DEVICES:
                                print(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                fwd = test_fwd_triton_torchref(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                bwd_x = test_bwdx_triton_torchref(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                bwd_w = test_bwdw_triton_torchref(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                print(fwd, bwd_x, bwd_w)
                                res.append((n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device, fwd, bwd_x, bwd_w))
    res = pd.DataFrame(res, columns=["n_trees", "max_heights", "n_time_steps", "max_delay", "block_size", "batch_size", "BLOCK_SIZE_N", "device", "fwd", "bwd_x", "bwd_w"])
    return res