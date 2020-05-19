import os
from shutil import copyfile


class RestoreMeta:
    """
    Restore specific meta files from a backup folder to another meta folder. Since the number of meta files is very high, if there is a mistake into one file, having to copy everything from a backup folder is slow and painful. Instead, this script copy only selected meta files. 
    """
    def __init__(self, meta_folder, save_folder, index_filename):
        """
        `meta_folder`: the path to the meta folder \n
        `save_folder`: the path to the save folder \n
        `index_filename`: the path to the file containing the filenames of the meta files to restore
        """
        self.meta_folder = meta_folder
        self.save_folder = save_folder
        self.filenames = []
        with open(index_filename, 'r') as f:
            for line in f:
                if "\n" in line:
                    self.filenames.append(line[:len(line)-1])
                else:
                    self.filenames.append(line)

    def restore(self):
        """
        Run the restauration
        """
        for file in self.filenames:
            image_info = os.path.splitext(file)[0].split("_")
            zoom = image_info[1]
            xtile = image_info[2]
            ytile = image_info[3]
            source = os.path.join(self.save_folder,
                                  zoom,
                                  xtile,
                                  file)
            dest = os.path.join(self.meta_folder,
                                zoom,
                                xtile,
                                file)

            copyfile(source, dest)


if __name__ == "__main__":
    meta_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta_save_2\\Real_World_Dataset_TMS_meta\\sat"
    save_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Real_World_Dataset_TMS_meta_copy\\sat"
    index_filename = "helipad_path_over_0.txt"

    restore_meta = RestoreMeta(meta_folder, save_folder, index_filename)

    restore_meta.restore()