#!/usr/bin/env bash
rsync -r -av -P  -e "ssh -i ~/.ssh/id_gpu_enst" jbitoun@gpu.enst.fr:Project/Dissertation/Helipad_DataBase/* /mnt/c/Users/jonas/Desktop/Helipad/Helipad_DataBase/
rsync -r -av -P  -e "ssh -i ~/.ssh/id_gpu_enst" jbitoun@gpu.enst.fr:Project/Dissertation/Helipad_DataBase_meta/* /mnt/c/Users/jonas/Desktop/Helipad/Helipad_DataBase_meta/

# meta original remote to local
rsync -r -av -P aisg@172.27.140.180:~/Documents/Jonas/Helipad/Helipad_DataBase_meta/Helipad_DataBase_meta_original/* /mnt/c/Users/jonas/Desktop/Helipad/Helipad_DataBase_meta/Helipad_DataBase_meta_original/

rsync -r -av -P aisg@172.27.140.180:~/Documents/Jonas/Detection/Detection_Dataset_meta/* /mnt/c/Users/jonas/Desktop/Detection/Detection_Dataset_meta/