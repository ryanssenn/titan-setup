import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    print(f"rank {rank}: initializing NCCL on cuda:{local_rank}", flush=True)

    dist.init_process_group(backend="nccl")

    if rank == 0:
        seed = torch.tensor([12345], device="cuda")
    else:
        seed = torch.tensor([0], device="cuda")

    print(f"rank {rank}: before broadcast seed={seed.item()}", flush=True)

    dist.broadcast(seed, src=0)

    print(f"rank {rank}: after broadcast seed={seed.item()}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
