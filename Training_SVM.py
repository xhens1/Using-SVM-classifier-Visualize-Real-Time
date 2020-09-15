from skimage.feature import hog
from sklearn.svm import LinearSVC
import joblib,glob,os,cv2
import numpy as np
from HogDB import DB


train_data = []
train_labels = []
pos_im_path = 'DATAIMAGE/positive/'
neg_im_path = 'DATAIMAGE/negative/'
model_path = 'models/models.dat'

# Positive HOG
for filename in glob.glob(os.path.join(pos_im_path, "*.png")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(1)

for filename in glob.glob(os.path.join(pos_im_path, "*.jpg")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(1)

# Negative HOG
for filename in glob.glob(os.path.join(neg_im_path, "*.jpg")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(0)

for filename in glob.glob(os.path.join(neg_im_path, "*.png")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(0)


train_data = np.float32(train_data)
train_labels = np.array(train_labels)

print('Loading Data........')
print('Training Data InFo :', len(train_data))
print('Training Data Count (1,0)', len(train_labels))
print("""SVM Classif Makeing...""")

print('Training...... Support Vector Machine')
model = LinearSVC().fit(train_data, train_labels)
joblib.dump(model, 'models/models.dat')
print('Model saved : {}'.format('models/models.dat'))