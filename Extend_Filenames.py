import cPickle as pickle
import random

with open('data/no_pain.pkl', 'rb') as f:
    data = pickle.load(f)

# For painfull frames
# names = data[0]
# final = []
# for i in names:
#     for au in range(6):
#         final.append('AU' + str(au) + i[i.rfind("/")+1:])

# For no pain frame
names = data[0]
final = []
for i in names:
    final.append(i[i.rfind("/")+1:])

random.shuffle(final)
random.shuffle(final)
random.shuffle(final)

with open('data/AUfilenames_no_pain.pkl', 'a') as f:
    pickle.dump(final, f)
