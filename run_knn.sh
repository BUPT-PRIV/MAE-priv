python -m torch.distributed.launch --nproc_per_node=8 tools/run_knn.py "$@"