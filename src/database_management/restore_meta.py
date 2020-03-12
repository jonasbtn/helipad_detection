import os
from shutil import copyfile


class RestoreMeta:

    def __init__(self, meta_folder, save_folder, index_filename):
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