Single GPU:
python main.py -a resnet18 /localscratch/sfreitas3/malnet-image-type

Multi GPU:
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:2222' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /localscratch/sfreitas3/malnet-image-type