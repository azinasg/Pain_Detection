#
# To Do List
# Aug Mode 8 and 9, later to stretching and ...
#

import cPickle as pickle
import sys

import random
import menpo.io as mio
import numpy as np
import deepdish as dd
from menpo.shape import PointCloud
from menpo.transform import Affine

sys.path.insert(0, '../VGG/keras-vggface-master/keras_vggface')
from vggface import VGGFace
from utils import preprocess_input


# ----------------------------------------------------------------------------------------------------------------------
# Defining Helper Functions for Data Preparation
# ----------------------------------------------------------------------------------------------------------------------
# Convert a Menpo Image.pixels(ndarray) to a Keras Image
def convert_img(img_menpo):
    '''
    Convert a Menpo image to a Keras image for Feature Extraction
    :param img_menpo: given menpo image
    :return: return normalized image with keras dtype
    '''
    tmp = np.swapaxes(img_menpo, 0, 2)
    tmp = np.swapaxes(tmp, 0, 1)
    tmp *= 255
    tmp = tmp.astype("float32")

    return tmp

# ----------------------------------------------------------------------------------------------------------------------
# Extract the features
# ----------------------------------------------------------------------------------------------------------------------
def get_features(img):
    '''
    Extract the VGG features (AVG Pooling of the last convolution layer in VGG Face Network).
    :param img: The input to the VGG networks
    :return: VGG features extracted from that image, shape (1,512).
    '''
    sample = convert_img(img)
    sample = np.expand_dims(sample, axis=0)
    sample = preprocess_input(sample, version=1)

    # Return Convolution Features
    return model_conv.predict(sample)

# ----------------------------------------------------------------------------------------------------------------------
# Read the image and landmakrs, Align the image and then crop it around the face bounding box
# ----------------------------------------------------------------------------------------------------------------------
def read_img(image_path, aug_mode, shear_x, shear_y, output_path, degree=0, align=True, crop_around_bb=True, save=True):
    '''
    Read the image in the given image path and apply the mentioned transformation steps.
    :param image_path: Path to the image
    :param aug_mode: Mode for augmrntation, can be any of the options below:

    :param align: If True align the images based on corner of the eyes
    :param crop_around_bb: If True crop the image arounf an extension of ground truth bounding box
            (adding 40% of the length to the top)
    :return: Return pixels of this image and landmarks after processing the desired transformation
    '''
    # Read the image with Menpo Library
    img = mio.import_image(image_path)
    land_path = image_path[:-4].replace('Images', 'AAM_landmarks') + "_aam.txt"
    with open(land_path) as file:
        tmp = np.array([[float(x) for x in line.split()] for line in file])

    # Swapping Columns (Y,X) -> (X,Y)
    result = np.zeros((66, 2))
    result[:, 0] = tmp[:, 1]
    result[:, 1] = tmp[:, 0]

    # Adding Landmarks
    P = PointCloud(result)
    img.landmarks.__setitem__('face_ibug_66_trimesh', P)

    # Align the images based on eye corners
    if aug_mode==0 and align:
        Leye_Rcorner = img.landmarks['face_ibug_66_trimesh'].points[39, :]
        Reye_Lcorner = img.landmarks['face_ibug_66_trimesh'].points[42, :]

        dx = Reye_Lcorner[1] - Leye_Rcorner[1]
        dy = Leye_Rcorner[0] - Reye_Lcorner[0]
        theta = np.arctan(dy / (1.0 * dx))

        img = img.rotate_ccw_about_centre(-theta, degrees=False, retain_shape=False)

    elif aug_mode==1 or aug_mode==8:
        # Rotate the image with a random degree less equal to the given degree
        img = img.rotate_ccw_about_centre(degree, degrees=True, retain_shape=False)

    elif aug_mode==2 or aug_mode==9:
        img = img.rotate_ccw_about_centre(-degree, degrees=True, retain_shape=False)

    elif aug_mode==3:
        img = img.mirror(axis=1, return_transform=False)

    elif aug_mode==4:
        shear_tr = Affine.init_from_2d_shear(shear_x, shear_y)
        img = img.transform_about_centre(shear_tr)

    elif aug_mode==5:
        img = img.mirror(axis=1, return_transform=False)
        shear_tr = Affine.init_from_2d_shear(shear_x, shear_y)
        img = img.transform_about_centre(shear_tr)

    elif aug_mode==6:
        img = img.mirror(axis=1, return_transform=False)
        img = img.rotate_ccw_about_centre(degree, degrees=True, retain_shape=False)

    elif aug_mode==7:
        img = img.mirror(axis=1, return_transform=False)
        img = img.rotate_ccw_about_centre(-degree, degrees=True, retain_shape=False)


    # Crop the bounding box based on lanrmakrs and add 40% to its top part
    if crop_around_bb:
        bb = img.landmarks['face_ibug_66_trimesh'].bounding_box()
        d_y = 0.4 * (bb.points[1, 0] - bb.points[0, 0])
        bb.points[0, 0] -= d_y
        bb.points[3, 0] -= d_y

        # Crop the images around the bounding box and resize it
        img = img.crop_to_pointcloud_proportion(bb, 0.5)
        img = img.resize(shape=(224, 224))

    if save:
        mio.export_image(img, output_path + 'AU_' + str(aug_mode) + '_' + img.path.name, overwrite=True)

    return img.pixels, img.landmarks['face_ibug_66_trimesh'].points

# ----------------------------------------------------------------------------------------------------------------------
# Read data
# ----------------------------------------------------------------------------------------------------------------------
def read_data(path, root, rt_deg, shear_x, shear_y, seq_len, output_path):

    with open(path, 'rb') as f:
        data = pickle.load(f)

    final_data = {}

    filenames = data[0]
    labels = data[1]

    for i in range(len(filenames)):

        for au in range(10):
            # Data Augmentation Parameters
            r_deg = random.uniform(0, rt_deg)
            shear_x = random.uniform(-shear_x, shear_x)
            shear_y = random.uniform(-shear_y, shear_y)

            f = filenames[i]
            pic_num = int(f[-7:-4])

            # Loop over the sequence
            for num in range(pic_num-seq_len+1, pic_num+1):

                f = filenames[i][:-7] + str(num).zfill(3) + '.png'
                key = 'AU'+str(au)+f[f.rfind("/")+1:-4]

                if key not in final_data.keys():
                    img_src = root + f
                    label = labels[i]
                    img, aam = read_img(img_src, au, shear_x, shear_y, output_path, r_deg)
                    feature = get_features(img)

                    final_data[key] = [feature[0,:], aam, label]

    return final_data

# ----------------------------------------------------------------------------------------------------------------------
# Model Params
# ----------------------------------------------------------------------------------------------------------------------

if __name__=='__main__':

    # Initialize Parameters
    root = '/Users/azinasgarian/Documents/Data/UNBC/Images'
    path = './data/high_pain.pkl'
    output_path = '/Users/azinasgarian/Desktop/test/'
    rotation_degree = 15
    seq_len = 15
    shear_x = 8
    shear_y = 8

    print "Building VGG Model ... "
    # pooling: None, avg or max
    model_conv = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    print "VGG Model is built! "

    print "Reading and extracting vgg features ..."
    Data = read_data(path, root, rotation_degree, shear_x, shear_y, seq_len, output_path)
    print "Features are extracted!"

    print "Saving data into .h5 file."
    dd.io.save('tmp.h5', Data)
    print "Data is saved!"

    print "All Done!"
