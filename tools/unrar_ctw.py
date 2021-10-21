import tarfile
import os
def unrar_files(folder,save_folder):
    for tar_ in os.listdir(folder):
        if ".tar" not in tar_:
            continue
        tar_name = os.path.join(folder, tar_)
        tar = tarfile.open(tar_name, "r")
        file_names = tar.getnames()
        for file_name in file_names:
            # print(file_name)
            tar.extract(file_name, save_folder)
        tar.close()
        # file.extractall(save_folder)
# unrar_files("/workspace/wanghao/datasets/CTW/images-test","/workspace/wanghao/datasets/CTW/test_images")
unrar_files("/workspace/wanghao/datasets/CTW/images-trainval","/workspace/wanghao/datasets/CTW/train_images")