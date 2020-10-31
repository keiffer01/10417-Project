# Segment the cropped images

import glob
from PIL import Image

# The length of the cropped image was predetermined to be divisible by 5.
num_segments = 5

# Get the cropped images path.
images_path = "../data/images_cropped"
genre_paths = glob.glob(images_path + "/*")

# Get the path that the segmented images will be placed.
new_images_path = "../data/images_segmented"
new_genre_paths = glob.glob(new_images_path + "/*")

zipped_genre_paths = list(zip(genre_paths, new_genre_paths))
for (genre_path, new_genre_path) in zipped_genre_paths:
  image_paths = glob.glob(genre_path + "/*")

  for image_path in image_paths:
    filename = image_path.split("/")[-1][0:-4]
    image = Image.open(image_path)
    length, height = image.size
    segment_size = length // num_segments

    for num_segment in range(num_segments):
      # Crop the image to a segment
      new_size = (num_segment*segment_size,
                  0,
                  (num_segment+1)*segment_size,
                  height
                 )
      segment = image.crop(new_size)

      # Save the segment
      new_filepath = new_genre_path + "/" + filename + "(" + str(num_segment+1) + ")" + ".png"
      segment.save(new_filepath)
