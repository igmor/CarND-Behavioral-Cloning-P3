import csv
import cv2
import numpy as np
import sklearn

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

print(len(samples))
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def preprocess_image(x):
	return (x - np.mean(x) / max(np.std(x), 1.0/len(x)))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        offset = 0
        while offset < num_samples:
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            while len(images) <= batch_size and offset < num_samples:
                batch_sample = samples[offset]
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
                name_right = './data/IMG/'+batch_sample[2].split('/')[-1]

                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                if center_image is None:
                   print(name)
                   offset+=1
                   continue
                center_angle = float(batch_sample[3])
                if abs(center_angle) < 0.05:
                   offset+=1
                   continue
                images.append(center_image)
                angles.append(center_angle)

                if len(images) == batch_size:
                   offset+=1
                   continue

                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped  = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

                # create adjusted steering measurements for the side camera images
                correction = 0.3 # this is a parameter to tune
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                if len(images) == batch_size or len(images) == batch_size-1:
                   offset+=1
                   continue

                # read in images from center, left and right cameras
                left_image = cv2.imread(name_left)
                left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                right_image = cv2.imread(name_right)
                right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                images.extend([left_image, right_image])
                angles.extend([steering_left, steering_right])

                offset+=1

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def generator2(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = './data/IMG/'+batch_sample[0].split('/')[-1]
                name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
                name_right = './data/IMG/'+batch_sample[2].split('/')[-1]

                center_image = cv2.imread(name_center)
                #center_image = preprocess_image(center_image)
                if center_image is None:
                  print(name_center)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                print(center_image.shape) 
                angles.append(center_angle)

                #center_image_flipped = np.fliplr(center_image)
                #center_angle_flipped  = -center_angle
                #images.append(center_image_flipped)
                #angles.append(center_angle_flipped)

                # create adjusted steering measurements for the side camera images
                correction = 0.25 # this is a parameter to tune
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                # read in images from center, left and right cameras
                #left_image = np.array(cv2.imread(name_left), dtype=np.float32)
                #left_image = preprocess_image(left_image)
                #right_image = np.array(cv2.imread(name_right), dtype=np.float32)
                #right_image = preprocess_image(right_image)
                #images.extend([left_image, right_image])
                #angles.extend([steering_left, steering_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def generator_r(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = images
            y_train = angles
            yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

def VGG_16():
    model = Sequential()
	
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x/255.0 - 0.5)))
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 5, 5, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.5))

#    model.add(Convolution2D(64, 3, 3, activation='relu'))
#    model.add(Convolution2D(64, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #model.add(Dropout(0.5))

    #model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))


#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(64, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(64, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(128, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(128, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(3, 3, 256, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(3, 3, 256, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(3, 3, 256, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
#    model.add(Dense(512))
    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dense(1))

    return model

# compile and train the model using the generator function
BATCH_SIZE=32
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = VGG_16()
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=15)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*4/BATCH_SIZE, validation_data=validation_generator,
	nb_val_samples=len(validation_samples), nb_epoch=6)

model.save('model.h5')

