import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def result(img_path, filename):


    #Initialize the Data Generator
    img_gen = ImageDataGenerator(rescale=1./255)

    # Create a dataframe that stores the path of the img uploaded by the user
    d = pd.DataFrame(data={"Brain MRI": img_path}, index=[0])

    test_img_gen = img_gen.flow_from_dataframe(dataframe=d, x_col="Brain MRI", target_size=(256, 256),
                                          color_mode='rgb', class_mode=None, seed=10)

    # Load the Model
    model = tf.keras.models.load_model("brain-mri-segmentation.h5", compile=False) 

    # Get the output image
    pred = model.predict(test_img_gen)
    img = np.squeeze(pred)

    # Get the path of folder where we want to save the image
    path = 'App/static/scans/results/' + filename + "_mask.jpeg"

    # Save the Image
    plt.imsave(path, img)
    return pred

