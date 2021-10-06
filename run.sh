./tools/dist_test.sh \
    work_dirs/remote/swb_22k_18e_16bs_all/remote_swb.py \
    work_dirs/remote/swb_22k_18e_16bs_all/latest.pth \
    2 \
    --eval-options imgfile_prefix="./work_imgs" \
    --format-only


array([0.02170333, 0.25092608, 0.60119722, 0.03025677, 0.01214302,
       0.01571181, 0.01009619, 0.02583056, 0.03213502])
