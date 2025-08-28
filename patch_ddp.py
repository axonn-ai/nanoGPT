import sys
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from mpi4py import MPI

# PCCL imports
sys.path.append("/ccs/home/keshprad/sc2025-pccl-reproducer-v2/")
from pccl.process_groups import ProcessGroups
from pccl.build_kernels import build
from pccl.all_reduce import all_reduce_2D

pg = None
def get_heir_pg():
    global pg 
    assert pg is not None, "did you call the patch_ddp function?"
    return pg

def is_global_pg(group):
    return (group is None) or (group == dist.group.WORLD) or (dist.get_world_size(group) == dist.get_world_size())

def pccl_all_reduce_hook(group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if is_global_pg(group):
        # define input tensor - get the flattened gradient tensor from the bucket
        input_tensor = bucket.buffer()
        # define output tensor
        output_tensor = torch.empty(input_tensor.size(0), 
                                    device=input_tensor.device, 
                                    dtype=input_tensor.dtype)
        all_reduce_2D(output_tensor, 
                    input_tensor, 
                    group=get_heir_pg(), 
                    async_op=False,
                    use_rh_and_rd=True,
                    use_pccl_cpp_backend=True)
        
        # Create a future and set the result
        future = torch.futures.Future()
        future.set_result(output_tensor)
        return future
    else:
        return allreduce_hook(group, bucket)

def patch_ddp(ddp_model):
    assert dist.is_initialized()
    assert dist.get_world_size()
    
    # auto-detect intra-node process group size
    intra_node_pg_size = torch.cuda.device_count()
    
    # build pccl
    if dist.get_rank() == 0:
        build()
        MPI.COMM_WORLD.Barrier()
    else:
        MPI.COMM_WORLD.Barrier()
        build()
    
    # setup process groups sub-communicators
    global pg 
    pg = ProcessGroups(intra_node_pg_size, 
                       dist.get_world_size() // intra_node_pg_size)

    # register communication hook with the DDP process group as state
    ddp_model.register_comm_hook(ddp_model.process_group, pccl_all_reduce_hook)