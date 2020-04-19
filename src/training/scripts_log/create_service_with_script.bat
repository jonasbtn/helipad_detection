call nssm install test_tensorflow C:\Users\AISG\Documents\Jonas\helipad_detection\src\training\test_tensorflow_run_2.bat
call nssm set test_tensorflow AppStdout C:\Users\AISG\Documents\Jonas\helipad_detection\src\training\test_tensorflow_stdout_2.log
call nssm set test_tensorflow AppStderr C:\Users\AISG\Documents\Jonas\helipad_detection\src\training\test_tensorflow_stderr_2.log
call nssm set test_tensorflow AppEnvironmentExtra python=C:\Users\AISG\Documents\Jonas\helipad_detection\helipad_detection_env\Scripts\python.exe