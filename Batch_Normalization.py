#
# We devide samples into 4 classes, no_pain(0), low_pain(1-3), medium_pain(4-6), high_pain(7-16)
#
#

import cPickle as pickle
import os
import random

# ----------------------------------------------------------------------------------------------------------------------
# Defining Helper Functions for Data Preparation
# ----------------------------------------------------------------------------------------------------------------------
def read_PSPI(PSPIPath):
    with open(PSPIPath) as file:
        pspi_score = float(file.readline())
    return pspi_score

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Params
# ----------------------------------------------------------------------------------------------------------------------
with open('data/medium_pain.pkl', 'rb') as f:
    data = pickle.load(f)


SRC = "/Users/azinasgarian/Documents/Data/UNBC/Images"
names = ['042-ll042', '043-jh043', '047-jl047', '048-aa048', '049-bm049', '052-dr052', '059-fn059', '064-ak064',
         '066-mg066', '080-bn080', '092-ch092', '095-tv095', '096-bg096', '097-gf097', '101-mg101', '103-jk103',
         '106-nm106', '107-hs107', '108-th108', '109-ib109', '115-jy115', '120-kz120', '121-vw121', '123-jh123',
         '124-dn124']


pain_score = [7,8,9,10,11,12,13,14,15,16]
labels = []
File_names = []


# ----------------------------------------------------------------------------------------------------------------------
# Read and process data
# ----------------------------------------------------------------------------------------------------------------------
for name in names:
    print name

    ROOT = SRC + '/' + name

    for root, dirs, filenames in os.walk(ROOT):
        if len(filenames) > 1:
            if 'DS_Store' in filenames[0]:
                del filenames[0]

            for f in filenames:
                img_src = root + '/' + f
                pspi_score = read_PSPI(img_src[:-4].replace('Images', 'Frame_Labels/PSPI') + '_facs.txt')

                if pspi_score in pain_score and int(f[-7:-4]) > 15:
                    File_names.append(img_src[img_src.find('Images') + 6:])
                    labels.append(pspi_score)

# ----------------------------------------------------------------------------------------------------------------------
# Sub-Sampling and Saving
# ----------------------------------------------------------------------------------------------------------------------
random.shuffle(File_names)

# Saving the Data
if not labels:
    labels = [0 for _ in range(len(File_names))]
print File_names

with open('data/high_pain.pkl', 'a') as f:
    pickle.dump((File_names, labels), f)

