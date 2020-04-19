call C:\Users\AISG\Anaconda3\Scripts\activate
call conda activate tf-gpu_1.13
call cd C:\Users\AISG\Documents\Jonas\Helipad\Mask_RCNN-master\
call python setup.py install
call cd C:\Users\AISG\Documents\Jonas\helipad_detection\src\training\
call python run_training.py >> helipad_train_log_3+_2.log
call C:\Users\AISG\nssm.exe stop helipad_train
