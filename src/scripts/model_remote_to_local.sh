#!/usr/bin/env bash
rsync -r -av -P  -e "ssh -i ~/.ssh/id_gpu_enst" jbitoun@gpu.enst.fr:Project/Dissertation/model/* /mnt/c/Users/jonas/Desktop/Helipad/model/


rsync -r -av -P  -e "ssh -i ~/.ssh/id_gpu_enst" jbitoun@gpu.enst.fr:Project/Dissertation/model/helipad_cfg20191124T1730/mask_rcnn_helipad_cfg_0013.h5 /mnt/c/Users/jonas/Desktop/Helipad/model/


rsync -r -av -P  aisg@172.27.140.96:~/Documents/Jonas/Helipad/model/helipad_cfg_6_aug4_3+20200103T1225/mask_rcnn_helipad_cfg_6_aug4_3+_0288.h5 /mnt/c/Users/jonas/Desktop/Helipad/model/
