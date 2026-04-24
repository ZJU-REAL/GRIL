export WANDB_MODE=offline
export TMPDIR=/temp
export RAY_TMPDIR=/temp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python train.py --config-name base