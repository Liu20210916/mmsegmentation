mkdir weights
python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth weights/swin_base_patch4_window12_384_22k.pth 

# 训练
gpus=2 # (GPU 数量)
./tools/dist_train.sh ./work_configs/remote/remote_swb.py $gpus
./tools/dist_train.sh ./work_configs/remote/remote_dl3pr101.py $gpus

python ./tools/ensemble_inferencer.py