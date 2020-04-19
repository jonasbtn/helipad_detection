import win32serviceutil
import win32service
import win32event
import servicemanager
import socket

from run_training import RunTraining


class AppServerSvc (win32serviceutil.ServiceFramework):
    _svc_name_ = "Helipad_Training"
    _svc_display_name_ = "Helipad_Training"

    def __init__(self,args):
        win32serviceutil.ServiceFramework.__init__(self,args)
        self.hWaitStop = win32event.CreateEvent(None,0,0,None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_,''))
        self.main()

    def main(self):
        root_folder = "../../../Helipad/Helipad_DataBase"
        root_meta_folder = "../../../Helipad/Helipad_DataBase_meta"
        model_folder = "../../../Helipad/model"
        include_augmented = False
        augmented_version = []

        train_categories = ["1", "2", "3", "5", "6", "8", "9"]
        test_categories = ["4", "7", "d", "u"]

        weights_filename = 'helipad_cfg_6_aug4_3+20200103T1225/mask_rcnn_helipad_cfg_6_aug4_3+_0288.h5'
        base_weights = 'mask_rcnn_coco.h5'

        predict_weights_filepath = 'helipad_cfg_7_aug123_all20200106T2012/mask_rcnn_helipad_cfg_7_aug123_all_0407.h5'

        run_training = RunTraining(root_folder,
                                   root_meta_folder,
                                   model_folder,
                                   base_weights,
                                   include_augmented=include_augmented,
                                   augmented_version=augmented_version,
                                   predict_weights_filepath=None,
                                   train_categories=train_categories,
                                   test_categories=None)

        print('Starting Training')
        run_training.run()
        print('Training Over')

        
if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(AppServerSvc)