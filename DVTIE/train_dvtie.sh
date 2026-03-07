# nohup bash train_dvtie.sh > outputs/training_dvtie_run1.log 2>&1 &
accelerate launch training_dvtie.py --config config/train.yaml
