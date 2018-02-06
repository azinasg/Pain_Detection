import menpo.io as mio
import numpy as np
import scipy.io as scio
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from menpo.shape import bounding_box


# ----------------------------------------------------------------------------------------------------------------------
# Defining Helper Functions for Data Preparation
# ----------------------------------------------------------------------------------------------------------------------
# Convert a Menpo Image.pixels(ndarray) to a Keras Image
def convert_img(img_menpo):
    tmp = np.swapaxes(img_menpo, 0, 2)
    tmp = np.swapaxes(tmp, 0, 1)
    tmp *= 255
    tmp = tmp.astype("float32")

    return tmp

# ----------------------------------------------------------------------------------------------------------------------
# Read the image and resize it to (224, 224)
# ----------------------------------------------------------------------------------------------------------------------
def read_img(path, filename, points):
    img = mio.import_image(path+filename)
    # img = img.crop([points[2], points[0]], [points[3], points[1]], constrain_to_boundary=False)

    # Evaluting the Bounding Box
    P = bounding_box([points[2], points[0]], [points[3], points[1]])

    # Adding extra 40% to top
    d_y = 0.3 * (P.points[1, 0] - P.points[0, 0])
    P.points[0, 0] -= d_y
    P.points[3, 0] -= d_y

    # Croping the image
    img = img.crop_to_pointcloud_proportion(P, 0.1)
    img = img.resize(shape=(224, 224))

    # Saving the image
    mio.export_image(img, './0_'+filename, overwrite=True)

    return img.pixels, img.landmarks['LJSON'].points

# ----------------------------------------------------------------------------------------------------------------------
# Read balanced data
# ----------------------------------------------------------------------------------------------------------------------
def read_data(path, BBs):
    data = {}
    for i in range(len(BBs)):
        filename = str(BBs[i][0][0][0])
        points = [BBs[i][0][1][0][0], BBs[i][0][2][0][0], BBs[i][0][3][0][0], BBs[i][0][4][0][0]]
        img, aam = read_img(path, filename, points)
        # Converting an image to the appropriate sample for extracting VGG features
        sample = convert_img(img)
        sample = np.expand_dims(sample, axis=0)
        sample = preprocess_input(sample, version=1)  # version=1 for VGG, version=2 for RESNET50, SENET50
        # Extracting the VGG features
        feature = model_conv.predict(sample)
        data[filename] = [feature[0,:], aam]
    return data

# ----------------------------------------------------------------------------------------------------------------------
# Model Params
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Initialization
    data_root = './data/'
    for side in ['frontal', 'profile']:
        BBs = scio.loadmat(side + '_bbox.mat')
        BBs = BBs[side + '_bbox']

        #Building the VGG Model
        model_conv = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        # Prepare the features
        Data = read_data(data_root, BBs)

        print 'Saving'
        scio.savemat('./Regina_' + side + '_features.mat', Data)
        np.save('./Regina_' + side + '_features.npy', Data)
        print 'Data is saved now!'
