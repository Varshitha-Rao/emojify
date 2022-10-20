from sre_parse import CATEGORIES
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import Sequential
import os

INIT_LR = 1e-4
EPOCHS = 15
BS = 32

DIRECTORY = r"C:\Users\HP\OneDrive\Desktop\Emojify\EmojifyDataset"
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category).replace("\\", "/")
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, grayscale = False, target_size = (48, 48, 0))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

#One Hot Encoding   
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#Convert into array format
data = np.array(data, dtype="uint8")
label = np.array(labels)

#Train Test Split

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify = labels)

# #Image Augumentation
aug = ImageDataGenerator(
     rotation_range = 20,
     width_shift_range= 0.2,
     height_shift_range = 0.2,
     brightness_range=[1, 1.2],
     shear_range = 0.1,
     zoom_range= 0.25,
     fill_mode='nearest',
 )

#Building the Model using CNN Architecture
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (48, 48, 3)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation = "relu"))
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Conv2D(128, (3,3), activation = "relu"))
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation= "relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation = "softmax"))

#Training 
opt = Adam(learning_rate= INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'ADAM',
    metrics = ['accuracy'],
)

model.fit(
    aug.flow(trainX, trainY, batch_size = BS), 
    steps_per_epoch = len(trainX)//BS,
    validation_data = (testX, testY),
    validation_batch_size = len(testX)//BS,
    verbose = 1, 
    epochs = EPOCHS
)

model.predict(
    testX,
    batch_size = BS,
    verbose = 1  
)



    