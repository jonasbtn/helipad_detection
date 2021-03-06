##

from helipad_detection.src.knn.knn_build_database import KNNBuildDatabase

def run_build_database():
    image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase\\Helipad_DataBase_original"
    meta_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Helipad_DataBase_meta\\Helipad_DataBase_meta_original"
    model_number = 7

    knn_build_database = KNNBuildDatabase(image_folder, meta_folder, model_number)

    knn_build_database.run()

##

from helipad_detection.src.knn.knn_training import KNNTraining

def run_training():

    knn_training = KNNTraining(nb_neighbors=2, nb_jobs=-1, test_size=0.25)

    knn_training.fit(knn_build_database.X, knn_build_database.y, mode="histogram", binary=True)

    knn_training.score()

    knn_training.save()
    
if __name__ == "__main__":
    pass
