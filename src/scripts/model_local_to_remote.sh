#!/usr/bin/env bash
rsync -r -av -P  -e "ssh -i ~/.ssh/id_gpu_enst" /mnt/c/Users/jonas/Desktop/Helipad/model/* jbitoun@gpu.enst.fr:Project/Dissertation/model/