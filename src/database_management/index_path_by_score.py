import os
import json
from tqdm import tqdm as tqdm


class IndexPathScore:
    """
    Create an index file of the images where an helipad has been detected by a model with a confidence above a certain score. An index help runs certain scripts faster by knowning exactly where are the bounding boxes. This reduces tremendously the number of files to load. The output file has the following format : `"helipad_path_over_{}_m{}.txt".format(score_threshold, model_number)`.
    """
    def __init__(self, meta_folder, model_number, score_threshold):
        """
        `meta_folder`: the folder containing the meta files. \n
        `model_number`: the number of the model wanted. \n
        `score_threshold`: the score threshold. All the images having all its bounding boxes below the score threshold are not added to the index. \n
        """
        self.meta_folder = meta_folder
        self.model_number = model_number
        self.score_threshold = score_threshold
        self.path_to_add = []
        self.output_filename = "helipad_path_over_{}_m{}.txt".format(score_threshold, model_number)

    def build_target_files(self):
        """
        Builds and returns a list of target filepaths.
        """
        target_files = []

        for subdirs, dirs, files in os.walk(self.meta_folder, topdown=True):
            for file in files:
                target_files.append(os.path.join(subdirs, file))

        return target_files

    def run(self):
        """
        Run the indexing.
        """
        target_files = self.build_target_files()
        for i in tqdm(range(len(target_files))):
            path = target_files[i]
            with open(path, 'r') as f:
                meta = json.load(f)
            if "predicted" not in meta:
                continue
            elif "model_{}".format(self.model_number) not in meta["predicted"]:
                continue
            elif not meta["predicted"][f'model_{self.model_number}']["helipad"]:
                continue
            above = False
            for score in meta["predicted"][f'model_{self.model_number}']["score"]:
                if score >= self.score_threshold:
                    above = True
            if above:
                self.path_to_add.append(os.path.basename(path))
            f.close()

    def write_output(self):
        """
        Save the index into a file. 
        """
        with open(self.output_filename, mode='wt', encoding='utf-8') as f:
            f.write('\n'.join(self.path_to_add))


if __name__ == "__main__":

    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta_save_2\\Real_World_Dataset_TMS_meta\\sat"
    model_number = 7
    score_threshold = 0

    index_path_score = IndexPathScore(meta_folder, model_number, score_threshold)

    index_path_score.run()

    index_path_score.write_output()


