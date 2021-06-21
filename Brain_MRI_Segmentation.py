#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[2]:


os.listdir("./lgg-mri-segmentation/kaggle_3m/")


# In[3]:


from glob import glob

data = "./lgg-mri-segmentation/kaggle_3m/"
train_imgs = []
mask_imgs = []
mask_imgs = glob(data + '*/*_mask*')

for i in mask_imgs:
    train_imgs.append(i.replace('_mask',''))


# In[4]:


len(train_imgs)


# In[5]:


len(mask_imgs)


# In[6]:


import cv2
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_images(image_path, mask_path):
  
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    fig, axis = plt.subplots(1,3, figsize=[11,13])

    axis[0].imshow(image)
    axis[0].set_title("Brain MRI")

    axis[1].imshow(mask)
    axis[1].set_title("Mask")

    axis[2].imshow(image)
    axis[2].imshow(mask, alpha = 0.5)
    axis[2].grid(False)
    axis[2].set_title("Brain MRI with mask")


# In[7]:


plot_images(train_imgs[1], mask_imgs[1])


# In[8]:


plot_images(train_imgs[45], mask_imgs[45])


# In[9]:


df = pd.DataFrame(data={"Brain MRI": train_imgs, "mask" : mask_imgs})


# In[10]:


df.head()


# In[11]:


from sklearn.model_selection import train_test_split
train_paths, test_paths = train_test_split(df, test_size=0.1)
train_paths, val_paths = train_test_split(train_paths, test_size=0.2)  


# In[12]:


print("No. of Training examples: " + str(len(train_paths)))
print("No. of Test examples: " + str(len(test_paths)))
print("No. of Validation examples: " + str(len(val_paths)))


# In[13]:


data_aug = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.03, shear_range=0.03,zoom_range=0.03,
                        horizontal_flip=True, fill_mode='nearest')


# In[14]:


BATCH_SIZE = 32
img_height = 256
img_width = 256
channels = 3


# In[15]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(rescale=1./255, **data_aug)
mask_gen = ImageDataGenerator(rescale=1./255, **data_aug)


# In[16]:


train_img_gen = img_gen.flow_from_dataframe(dataframe=train_paths, x_col="Brain MRI", batch_size=BATCH_SIZE, color_mode='rgb',
                                            target_size=(img_height, img_width), class_mode=None, seed = 10)
train_mask_gen = mask_gen.flow_from_dataframe(dataframe=train_paths, x_col="mask", batch_size=BATCH_SIZE, target_size=(img_height, img_width),
                                              color_mode='grayscale', class_mode=None, seed=10)


# In[17]:


val_img_gen = img_gen.flow_from_dataframe(dataframe=val_paths, x_col="Brain MRI", target_size=(img_height, img_width), batch_size=BATCH_SIZE,
                                          color_mode='rgb', class_mode=None, seed=10)
val_mask_gen = mask_gen.flow_from_dataframe(dataframe=val_paths, x_col="mask", batch_size=BATCH_SIZE, target_size=(img_height, img_width),
                                              color_mode='grayscale', class_mode=None, seed=10)


# In[18]:


test_img_gen = img_gen.flow_from_dataframe(dataframe=test_paths, x_col="Brain MRI", target_size=(img_height, img_width), batch_size=BATCH_SIZE,
                                          color_mode='rgb', class_mode=None, seed=10)
test_mask_gen = mask_gen.flow_from_dataframe(dataframe=test_paths, x_col="mask", batch_size=BATCH_SIZE, target_size=(img_height, img_width),
                                              color_mode='grayscale', class_mode=None, seed=10)


# ## U-Net Xception-style model

# In[19]:


# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.optimizers import Adam


# In[20]:


d_filters = [64, 128]
u_filters = [128, 64, 32]


# In[21]:


def unet_model(img_size, num_classes=1):

    inputs = keras.Input(shape = img_size + (3,))

    x = Conv2D(32, kernel_size=(3,3), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    prev_block_activation = x

    for filters in d_filters:
        x = Activation("relu")(x)
        x = Conv2D(filters=filters, kernel_size=(3,3), padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        X = SeparableConv2D(filters=filters, kernel_size=(3,3), padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)

        res = Conv2D(filters, kernel_size=(1,1), strides=2, padding="same")(prev_block_activation)
        x = layers.add([x, res])
        prev_block_activation = x

    # Up-Sampling
    for filters in u_filters:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        res = UpSampling2D(2)(prev_block_activation)
        res = Conv2D(filters, kernel_size=(1,1), padding="same")(res)
        x = layers.add([x, res])
        prev_block_activation = x

    outputs = Conv2D(num_classes, (1,1), activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


# In[22]:


keras.backend.clear_session()

model = unet_model((img_height, img_width))
model.summary()


# In[23]:


keras.utils.plot_model(model, show_shapes=True)


# In[24]:


model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])


# In[25]:


def data_iterator(image_gen, mask_gen):
    for img, mask in zip(image_gen, mask_gen):
        yield img, mask


# In[26]:


train_generator = data_iterator(train_img_gen, train_mask_gen)
valid_generator = data_iterator(val_img_gen, val_mask_gen)


# In[27]:


# Callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
model_callbacks = [
             ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5, verbose=1),
             ModelCheckpoint("brain-mri-segmentation.h5", save_best_only=True)
]


# In[46]:


model.fit(train_generator, epochs=15, callbacks=model_callbacks, validation_data=valid_generator, batch_size=BATCH_SIZE, 
          steps_per_epoch=train_img_gen.n/BATCH_SIZE, validation_steps=val_img_gen.n/BATCH_SIZE,verbose=1)


# In[28]:


model.load_weights("brain-mri-segmentation.h5")


# In[29]:


test_generator = data_iterator(test_img_gen, test_mask_gen)


# In[31]:


results = model.evaluate(test_generator, steps=test_img_gen.n/BATCH_SIZE, verbose=1)


# ## Plotting the result

# In[39]:


def plot_test_res(n_res=5):
    
    for i in range(n_res):
        
        idx = np.random.randint(0, len(test_paths))
        img = cv2.imread(test_paths["Brain MRI"].iloc[idx])
        img = cv2.resize(img, (img_height, img_width))
        img = img/255
        img = img[np.newaxis, :, :, :]
        prediction = model.predict(img)
        mask = cv2.imread(test_paths["mask"].iloc[idx])
        
        plt.figure(figsize=(11,13))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.title("Test Image")
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(mask))
        plt.title("Test Image Mask")
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(prediction))
        plt.title("Predicted Mask")
        plt.show()


# In[44]:


plot_test_res(n_res=7)

