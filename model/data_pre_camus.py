# process CAMUS data (ge vivide95 a2c a4c single frames (ed & es))
import cv2
import os
import SimpleITK as sitk
import numpy as np

img_size = 128


# def read_mhd(path, is_gt=False):
#     image = sitk.ReadImage(path)
#     image = sitk.GetArrayFromImage(image)
#     image = np.squeeze(image[0])
#     if is_gt is True:
#         image[image != 1] = 0  # just keep lv cavity, thus 0 is background and 1 is lv
#         image = cv2.resize(image, (img_size, img_size))
#     else:
#         image = cv2.resize(image, (img_size, img_size))
#         image = data_norm(image)
#     return image


def data_norm(inputs):
    inputs = np.array(inputs, dtype=np.float32)
    inputs  = inputs - np.mean(inputs)
    outputs = inputs / (np.std(inputs) + 1e-12)
    return outputs


dir_path = 'F:/data/CAMUS/testing'
# a2c_ims=np.zeros((0,2,128,128))
a2c_ims, a2c_gts = [], []
a4c_ims, a4c_gts = [], []

for p_dir in os.listdir(dir_path):
    p_dir_path = dir_path + '/' + p_dir

    a2c_eds_ims, a2c_eds_gts = [], []
    a4c_eds_ims, a4c_eds_gts = [], []
    for file in os.listdir(p_dir_path):
        file_path = p_dir_path + '/' + file

        if file.split('.')[1] == 'mhd':  # find .mhd

            # if file.split('.')[0].split('_')[1] == '2CH' and file.split('.')[0].split('_')[-1] == 'ED':  # find a2c ed
            #     a2c_eds_ims.append(read_mhd(file_path))
            if file.split('.')[0].split('_')[1] == '2CH' and file.split('.')[0].split('_')[-1] == 'ES':  # find a2c es
                a2c_eds_ims.append(read_mhd(file_path))
            # elif file.split('.')[0].split('_')[1] == '4CH' and file.split('.')[0].split('_')[-1] == 'ED':  # find a4c ed
            #     a4c_eds_ims.append(read_mhd(file_path))
            elif file.split('.')[0].split('_')[1] == '4CH' and file.split('.')[0].split('_')[-1] == 'ES':  # find a4c es
                a4c_eds_ims.append(read_mhd(file_path))
            # ----------------------------------------------------------------------------------------------------------
            # elif file.split('.')[0].split('_')[1] == '2CH' and file.split('.')[0].split('_')[-2] == 'ED' and \
            #         file.split('.')[0].split('_')[-1] == 'gt':  # find a2c ed gt
            #     a2c_eds_gts.append(read_mhd(file_path, is_gt=True))
            elif file.split('.')[0].split('_')[1] == '2CH' and file.split('.')[0].split('_')[-2] == 'ES' and \
                    file.split('.')[0].split('_')[-1] == 'gt':  # find a2c es gt
                a2c_eds_gts.append(read_mhd(file_path, is_gt=True))
            # elif file.split('.')[0].split('_')[1] == '4CH' and file.split('.')[0].split('_')[-2] == 'ED' and \
            #         file.split('.')[0].split('_')[-1] == 'gt':  # find a4c ed gt
            #     a4c_eds_gts.append(read_mhd(file_path, is_gt=True))
            elif file.split('.')[0].split('_')[1] == '4CH' and file.split('.')[0].split('_')[-2] == 'ES' and \
                    file.split('.')[0].split('_')[-1] == 'gt':  # find a4c es gt
                a4c_eds_gts.append(read_mhd(file_path, is_gt=True))
    a2c_ims.append(np.array(a2c_eds_ims)), a2c_gts.append(a2c_eds_gts)
    a4c_ims.append(np.array(a4c_eds_ims)), a4c_gts.append(a4c_eds_gts)

a2c_ims, a4c_ims = np.array(a2c_ims, dtype=np.float32), np.array(a4c_ims, dtype=np.float32)
a2c_gts, a4c_gts = np.array(a2c_gts, dtype=np.float32), np.array(a4c_gts, dtype=np.float32)

print(a2c_ims.shape, a4c_ims.shape)
print(a2c_gts.shape, a4c_gts.shape)

np.save('F:/data/CAMUS/cam_ES/ims_a2c.npy', a2c_ims)
np.save('F:/data/CAMUS/cam_ES/gts_a2c.npy', a2c_gts)
np.save('F:/data/CAMUS/cam_ES/ims_a4c.npy', a4c_ims)
np.save('F:/data/CAMUS/cam_ES/gts_a4c.npy', a4c_gts)
