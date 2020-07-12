import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

scale_factors = [2, 3, 4]
for scale_factor in scale_factors:
  new_image_dir = "./B_{}".format(scale_factor)
  if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)
  print("Now deal {}x images...".format(scale_factor))
  raw_img_dir = "./B"

  filename = os.listdir(raw_img_dir)
  for img in filename:
    image = Image.open(os.path.join(raw_img_dir, img)).convert("RGB")
    size = image.size
    image_size = image.resize((scale_factor * size[0], scale_factor * size[1]), Image.ANTIALIAS)
    image_size.save(os.path.join(new_image_dir, img))