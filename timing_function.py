from imports import *
import time

from torch_function import block_sparse_conv_1d_autograd
from gen_data import generate_inputs

sys.path.append("..")
from jphydro.routing.block_sparse_conv import BlockSparseCausalConv
from jphydro.routing.kernel_aggregator_irf import RoutingIRFAggregator

def time_cuda_function(func, args, n_warmup=10, n_iters=100):
    # Warmup runs
    for _ in range(n_warmup): func(*args)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    # Timing runs
    start_event.record()
    for _ in range(n_iters): func(*args)
    end_event.record()
    
    # Time computation
    torch.cuda.synchronize()    
    elapsed_time = start_event.elapsed_time(end_event)
    return elapsed_time / n_iters

def time_fwd_triton(n_trees = 1,
                     max_heights = 3,
                     n_time_steps = 365,
                     max_delay = 10,
                     block_size = 32,
                     batch_size = 1,
                     BLOCK_SIZE_N = 64,
                     device = "cuda:0"):
    """
    Run forward pass tests and time two implementations.
    """
    BLOCK_SIZE_M = block_size
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size)
    x = x.to(device)
    k = k.to(device)
    
    return time_cuda_function(block_sparse_conv_1d_autograd,
                              args=(x, k.block_indices, k.block_values, BLOCK_SIZE_M, BLOCK_SIZE_N),
                              n_warmup=10, n_iters=100)

def time_bwd_triton(n_trees=1,
                    max_heights=3,
                    n_time_steps=365,
                    max_delay=10,
                    block_size=32,
                    batch_size=1,
                    BLOCK_SIZE_N=64,
                    device="cuda:0"):
    """
    Time the backward pass for the autograd-style implementation.
    """
    BLOCK_SIZE_M = block_size
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size)
    x = x.to(device)
    k = k.to(device)
    # Enable gradient tracking on the input.
    
    out = block_sparse_conv_1d_autograd(x, k.block_indices, k.block_values, BLOCK_SIZE_M, BLOCK_SIZE_N)
    loss = (out**2).mean()
    backward_step_autograd = lambda:loss.backward(retain_graph=True)
    
    return time_cuda_function(backward_step_autograd, 
                              args=(), n_warmup=10, n_iters=100)


def time_fwd_th(n_trees = 1,
                     max_heights = 3,
                     n_time_steps = 365,
                     max_delay = 10,
                     block_size = 32,
                     batch_size = 1,
                     BLOCK_SIZE_N = 64,
                     device = "cuda:0"):
    """
    Run forward pass tests and time two implementations.
    """
    BLOCK_SIZE_M = block_size
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size)
    x = x.to(device)
    k = k.to(device)

    layer = BlockSparseCausalConv(k).to(device)
    return time_cuda_function(layer, args=(x,), n_warmup=10, n_iters=100)

def time_bwd_th(n_trees=1,
                max_heights=3,
                n_time_steps=365,
                max_delay=10,
                block_size=32,
                batch_size=1,
                BLOCK_SIZE_N=64,
                device="cuda:0"):
    """
    Time the backward pass for the BlockSparseCausalConv module.
    """
    BLOCK_SIZE_M = block_size
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size)
    x = x.to(device)
    k = k.to(device)
    x.requires_grad_()

    # Create the layer and move it to the correct device.
    layer = BlockSparseCausalConv(k).to(device)
    out  = layer(x)
    loss = (out**2).mean()
    backward_step_th = lambda: loss.backward(retain_graph=True)
        
    return time_cuda_function(
        backward_step_th,
        args=(),
        n_warmup=10,
        n_iters=100
    )

def time_all_fwd(
        N_TREES     = [1, 5],
        MAX_HEIGHTS = [3, 5],
        MAX_DELAYS  = [2, 10],
        BLOCK_SIZE  = [16, 32],
        BATCH_SIZES = [1, 8],
        TIME_STEPS  = [14, 365],
        device="cuda:0"
    ):
    BLOCK_SIZE_N = 64
    res = []
    for n_time_steps in TIME_STEPS:
        for n_trees in N_TREES:
            for max_heights in MAX_HEIGHTS:
                for max_delay in MAX_DELAYS:
                    for block_size in BLOCK_SIZE:
                        for batch_size in BATCH_SIZES:
                                triton = time_fwd_triton(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                th = time_fwd_th(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                res.append((n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device, triton, th))
    res = pd.DataFrame(res, columns=["n_trees", "max_heights", "n_time_steps", "max_delay", "block_size", "batch_size", "BLOCK_SIZE_N", "device", "triton", "th"])
    return res

def time_all_bwd(
    N_TREES     = [1, 5],
    MAX_HEIGHTS = [3, 5],
    MAX_DELAYS  = [2, 10],
    BLOCK_SIZE  = [16, 32],
    BATCH_SIZES = [1, 8],
    TIME_STEPS  = [14, 365],
    BLOCK_SIZE_N = 64,
    device="cuda:0"
    ):
    """
    Run backward pass timings for various configurations and return results in a DataFrame.
    """
    BLOCK_SIZE_N = 64
    res = []
    for n_time_steps in TIME_STEPS:
        for n_trees in N_TREES:
            for max_heights in MAX_HEIGHTS:
                for n_time_steps in TIME_STEPS:
                    for max_delay in MAX_DELAYS:
                        for block_size in BLOCK_SIZE:
                            for batch_size in BATCH_SIZES:
                                bwd_triton = time_bwd_triton(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                bwd_th = time_bwd_th(n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device)
                                res.append((n_trees, max_heights, n_time_steps, max_delay, block_size, batch_size, BLOCK_SIZE_N, device, bwd_triton, bwd_th))
    res = pd.DataFrame(res, columns=["n_trees", "max_heights", "n_time_steps", "max_delay", "block_size", "batch_size", "BLOCK_SIZE_N", "device", "bwd_triton", "bwd_th"])
    return res


def time_fwd_irf_agg( n_trees = 1,
                      max_heights = 3,
                      n_time_steps = 365,
                      max_delay = 10,
                      block_size = 32,
                      batch_size = 1,
                      BLOCK_SIZE_N = 64,
                      device = "cuda:0"):
    """
    Run forward pass tests and time two implementations.
    """
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size)
    delays = torch.FloatTensor([g.nodes[n]["tau"] for n in g.nodes]).to(device).requires_grad_()
    agg = RoutingIRFAggregator(g, None, max_delay=max_delay, block_size=block_size).to(device)
    
    autograd_forward = lambda delays: agg(delays)
    
    return time_cuda_function(
        autograd_forward,
        args=(delays,),
        n_warmup=10,
        n_iters=100
    )

def time_bwd_irf_agg(n_trees=1,
                     max_heights=10,
                     n_time_steps=365,
                     max_delay=10,
                     block_size=32,
                     batch_size=1,
                     BLOCK_SIZE_N=64,
                     device="cuda:0"):
    """
    Time the backward pass for the autograd-style implementation.
    """
    g, x, k = generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size)
    delays = torch.FloatTensor([g.nodes[n]["tau"] for n in g.nodes]).to(device).requires_grad_()
    agg = RoutingIRFAggregator(g, None, max_delay=max_delay, block_size=block_size).to(device)

    k = agg(delays)
    loss = k.block_values.sum()

    backward_step_autograd = lambda :loss.backward(retain_graph=True)
        
    return time_cuda_function(
        backward_step_autograd,
        args=(),
        n_warmup=10,
        n_iters=100
    )