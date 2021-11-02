./tools/dist_test.sh \
    work_dirs/remote/swb_22k_18e_16bs_all/remote_swb.py \
    work_dirs/remote/swb_22k_18e_16bs_all/latest.pth \
    2 \
    --eval-options imgfile_prefix="./work_imgs" \
    --format-only


array([0.02170333, 0.25092608, 0.60119722, 0.03025677, 0.01214302,
       0.01571181, 0.01009619, 0.02583056, 0.03213502])

docker run \
    -it \
    --gpus=all \
    -v /home/zhaoxun/codes/mmsegmentation/data/remote2/test:/input_path \
    -v /home/zhaoxun/codes/mmsegmentation/work_imgs:/output_path \
    -v /home/zhaoxun/codes/mmsegmentation:/data \
    --shm-size 8G \
    mmseg:latest \
    /bin/bash
    python /workspace/run.py /input_path /output_path


###
# docker install
apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim libgl1-mesa-glx \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
conda clean --all
pip install mmcv-full==1.3.10 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html
git clone https://github.com/CarnoZhao/mmsegmentation.git
cd mmsegmentation
git checkout v0.17.0.base
pip install -r requirements.txt
pip install --no-cache-dir -e .

# create submission
target=/workspace/work_dirs/remote2/swb384_22k_1x_10bs2acc_all
subtarget=$(echo $target | awk 'BEGIN{FS = "/"}{print $NF}')
mkdir -p $target
cd $target
cp /data/work_dirs/remote2/$subtarget/epoch_12.pth ./
cp /data/work_dirs/remote2/$subtarget/remote2_swb.py ./
cd /workspace
vim run.py
