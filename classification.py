import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Parameters
path = "myData"
labelFile = 'labels.csv'
batch_size_val = 50
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

# Importing the Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Read CSV
data = pd.read_csv(labelFile)
print(data.shape)


# Preprocessing
def grayscale(picture):
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    return picture


def equalize(picture):
    picture = cv2.equalizeHist(picture)
    return picture


def preprocessing(picture):
    picture = grayscale(picture)
    picture = equalize(picture)
    picture = picture / 255
    return picture


X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
imgTest = X_train[random.randint(0, len(X_train) - 1)]

# Adding a depth of 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Augmentation of Images to make it more generic
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train,
                       batch_size=20)
X_batch, y_batch = next(batches)
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


# CNN Model
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter,
                      input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val), steps_per_epoch=200, epochs=1,
                              validation_data=(X_validation, y_validation), shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Testing the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Test Image Parameters
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX


# List of Classes
def getClassName(classno):
    if classno == 0:
        return 'Speed Limit 20 km/h'
    elif classno == 1:
        return 'Speed Limit 30 km/h'
    elif classno == 2:
        return 'Speed Limit 50 km/h'
    elif classno == 3:
        return 'Speed Limit 60 km/h'
    elif classno == 4:
        return 'Speed Limit 70 km/h'
    elif classno == 5:
        return 'Speed Limit 80 km/h'
    elif classno == 6:
        return 'End of Speed Limit 80 km/h'
    elif classno == 7:
        return 'Speed Limit 100 km/h'
    elif classno == 8:
        return 'Speed Limit 120 km/h'
    elif classno == 9:
        return 'No passing'
    elif classno == 10:
        return 'No passing for vehicles over 3.5 metric tons'
    elif classno == 11:
        return 'Right-of-way at the next intersection'
    elif classno == 12:
        return 'Priority road'
    elif classno == 13:
        return 'Yield'
    elif classno == 14:
        return 'Stop'
    elif classno == 15:
        return 'No vehicles'
    elif classno == 16:
        return 'Vehicles over 3.5 metric tons prohibited'
    elif classno == 17:
        return 'No entry'
    elif classno == 18:
        return 'General caution'
    elif classno == 19:
        return 'Dangerous curve to the left'
    elif classno == 20:
        return 'Dangerous curve to the right'
    elif classno == 21:
        return 'Double curve'
    elif classno == 22:
        return 'Bumpy road'
    elif classno == 23:
        return 'Slippery road'
    elif classno == 24:
        return 'Road narrows on the right'
    elif classno == 25:
        return 'Road work'
    elif classno == 26:
        return 'Traffic signals'
    elif classno == 27:
        return 'Pedestrians'
    elif classno == 28:
        return 'Children crossing'
    elif classno == 29:
        return 'Bicycles crossing'
    elif classno == 30:
        return 'Beware of ice/snow'
    elif classno == 31:
        return 'Wild animals crossing'
    elif classno == 32:
        return 'End of all speed and passing limits'
    elif classno == 33:
        return 'Turn right ahead'
    elif classno == 34:
        return 'Turn left ahead'
    elif classno == 35:
        return 'Ahead only'
    elif classno == 36:
        return 'Go straight or right'
    elif classno == 37:
        return 'Go straight or left'
    elif classno == 38:
        return 'Keep right'
    elif classno == 39:
        return 'Keep left'
    elif classno == 40:
        return 'Roundabout mandatory'
    elif classno == 41:
        return 'End of no passing'
    elif classno == 42:
        return 'End of no passing by vehicles over 3.5 metric tons'


# Read Test Image
imgOriginal = cv2.imread("my_sign_images/004.jpg")

# Preprocess Test Image
img = np.asarray(imgOriginal)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
cv2.namedWindow('Preprocessed Image', cv2.WINDOW_NORMAL)
cv2.imshow("Preprocessed Image", img)
cv2.waitKey(0)
cv2.destroyWindow('Preprocessed Image')
img = img.reshape(1, 32, 32, 1)
cv2.putText(imgOriginal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(imgOriginal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

# Prediction
predictions = model.predict(img)
classIndex = np.argmax(model.predict(img), axis=-1)
probabilityValue = np.amax(predictions)
if probabilityValue > threshold:
    print(getClassName(classIndex))
    cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 0), 2,
                cv2.LINE_AA)
cv2.namedWindow('Final Image', cv2.WINDOW_NORMAL)
cv2.imshow('Final Image', imgOriginal)
cv2.waitKey(0)
cv2.destroyWindow('Preprocessed Image')

print(getClassName(classIndex))

