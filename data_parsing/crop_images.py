# Crops the white border from the image dataset

import glob
from PIL import Image

# Cropped image size that removes the white border, determined by hand
# (left, top, right, bottom)
newSize = (54, 35, 389, 252)

images_path = "../data/images_original"
genre_paths = glob.glob(images_path + "/*")

for genre_path in genre_paths:
  image_paths = glob.glob(genre_path + "/*")

  for image_path in image_paths:
    image = Image.open(image_path)
    newImage = image.crop(newSize)
    newImage.save(image_path)
