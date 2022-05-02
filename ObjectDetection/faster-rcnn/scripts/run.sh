nohup python -m visdom.server &

CUDA_VISIBLE_DEVICES=0
python train.py train --env='fasterrcnn' --plot-every=100