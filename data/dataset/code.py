# from keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array, ImageDataGenerator
# Total Generated number
total_number = 100

data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                              zoom_range=0.2, horizontal_flip=True, vertical_flip=True, rotation_range=2, brightness_range=[0.2, 1.0])

# Create image to tensor
img = load_img("VWperfect.png", grayscale=False)
arr = img_to_array(img)
tensor_image = arr.reshape((1, ) + arr.shape)
imgnum = 1
for i, _ in enumerate(data_gen.flow(x=tensor_image,
                                    batch_size=1,
                                    save_to_dir="generateddataset/VW",
                                    save_prefix="",
                                    save_format=".png")):
    if i > total_number:
        break
