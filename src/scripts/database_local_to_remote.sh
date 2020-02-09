#!/usr/bin/env bash
rsync -r -av -P  -e "ssh -i ~/.ssh/id_gpu_enst" /mnt/c/Users/jonas/Desktop/Helipad/Helipad_DataBase/* jbitoun@gpu.enst.fr:Project/Dissertation/Helipad_DataBase/
rsync -r -av -P  -e "ssh -i ~/.ssh/id_gpu_enst" /mnt/c/Users/jonas/Desktop/Helipad/Helipad_DataBase_meta/* jbitoun@gpu.enst.fr:Project/Dissertation/Helipad_DataBase_meta/


rsync -r -av -P  /mnt/c/Users/jonas/Desktop/Helipad/Helipad_DataBase_meta/Helipad_DataBase_meta_original/* aisg@172.27.142.189:~/Documents/Jonas/Helipad/Helipad_DataBase_meta/Helipad_DataBase_meta_original/
