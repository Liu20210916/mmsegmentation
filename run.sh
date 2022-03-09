./tools/dist_test.sh \
    work_dirs/binzhou/convx_l_2x_ft_all/binzhou_convx_l_ft.py \
    work_dirs/binzhou/convx_l_2x_ft_all/epoch_24.pth \
    4 \
    --eval-options imgfile_prefix="./work_imgs" \
    --format-only
mkdir results
python -c "import os; import cv2; from skimage import io; [io.imsave(os.path.join('./results', x.replace('GF', 'LT').replace('png', 'tif')), cv2.imread('./work_imgs/' + x, cv2.IMREAD_UNCHANGED) + 1) for x in os.listdir('./work_imgs')]" > /dev/null
cd results; zip results.zip -r -9 *.tif; mv results.zip ../data/binzhou/; cd ..; rm results -rf

./tools/dist_test.sh \
    work_dirs/tamper/convx_l_6x_dice_aug1/tamper_convx_l.py \
    work_dirs/tamper/convx_l_6x_dice_aug1/epoch_72.pth \
    2 \
    --options \
    model.test_cfg.binary_thres=0.4 \
    data.test.pipeline.1.flip=False \
    --format-only \
    --eval-options imgfile_prefix="./data/tamper/test/images" \
    data.test.img_dir="train2/img" \
    data.test.ann_dir="train2/msk" \
    --eval mIoU mFscore \

python -c "import glob, cv2; [cv2.imwrite(_, cv2.imread(_, cv2.IMREAD_UNCHANGED) * 255) for _ in glob.glob('./data/tamper/test/images/*')]"; cd ./data/tamper/test; zip images.zip -9 -r ./images

python -c "import glob, cv2; print(sum([(cv2.imread(_, cv2.IMREAD_UNCHANGED) == 0).all() for _ in glob.glob('./data/tamper/test/images/*')]))"