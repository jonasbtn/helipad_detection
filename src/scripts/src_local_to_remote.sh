#!/usr/bin/env bash

rsync -r -av -e "ssh -i ~/.ssh/id_gpu_enst" /mnt/c/Users/jonas/Google\ Drive/NUS/Dissertation/src/* jbitoun@gpu.enst.fr:Project/Dissertation/src/

rsync -r -av -e "ssh -i ~/.ssh/id_gpu_enst" /mnt/c/Users/jonas/Desktop/Mask_RCNN-master/* jbitoun@gpu.enst.fr:Project/Dissertation/Mask_RCNN-master/

# local to 1080
rsync -r -av /mnt/c/Users/jonas/Google\ Drive/NUS/Dissertation/src/* aisg@172.27.140.180:~/Documents/Jonas/Helipad/src/


rsync -r -av /mnt/c/Users/jonas/Google\ Drive/NUS/Dissertation/src/* aisg@172.17.139.115:~/Documents/Jonas/src/

# local to 2080
pscp -r "Google Drive"\NUS\Dissertation\src\* AISG@172.17.139.11:C:\Users\AISG\Documents\Jonas\Helipad\src\

# remote to local
rsync -r -av aisg@172.27.140.180:~/Documents/Jonas/Helipad/src/benchmark_model* /mnt/c/Users/jonas/Google\ Drive/NUS/Dissertation/src/