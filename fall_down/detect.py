import torch
import torchvision
import argparse

import cv2
import numpy as np
import sys

sys.path.append('./')
import coco_names
import random
import os
import torchvision.transforms as standard_transforms
from PIL import Image
from  matplotlib import pyplot as plt


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)

@torch.no_grad()
def main():
    # args = get_args()
    image_path = './fall/'  # image path
    model_name = 'keypointrcnn_resnet50_fpn'
    dataset = 'coco'  # model pretrain dataset
    score = 0.8 # objectness score threshold
    asp = 0.8
    
    num_classes = 2  # 91
    names = coco_names.names

    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[model_name](num_classes=num_classes, pretrained=True)
    model = model.cuda()
    model.eval()

    image_count = 0
    image_list = os.listdir(image_path)
    for image_name in image_list:
      input = []  
      image_dir = os.path.join(image_path, image_name)
      src_img = cv2.imread(image_dir)
      print(image_name, src_img.shape)
      img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
      img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
      input.append(img_tensor)
      out = model(input)
      boxes = out[0]['boxes']
      labels = out[0]['labels']
      scores = out[0]['scores']
      # masks = out[0]['masks']
      keypoints = out[0]['keypoints']

      count = 0
      for index, idx in enumerate(range(boxes.shape[0])):
          if scores[idx] >= score:
              x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
              # print(names[str(labels[idx].item())], scores[idx])
              if names[str(labels[idx].item())] != 'person':
                continue
              w, h = x2 - x1, y2 - y1
              # print(w,h)
              
              name = names.get(str(labels[idx].item()))
              cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
              cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))
                          
              one = []
              two = []
              three = []
              four = []
              head_y = []
              foot_y = []
              for index__, point in enumerate(keypoints[idx]):
                x, y, vis = point
                x, y, vis = x.item(), y.item(), vis.item()
                if vis == 0:
                  continue
                if (index__ == 5) or (index__ == 6) or (index__ == 11) or (index__ == 12) or (index__ == 13) or (index__ == 14):
                  if (index__ == 5) or (index__ == 6):
                    one.append((int(x), int(y)))
                  if (index__ == 11) or (index__ == 12):
                    two.append((int(x), int(y)))
                  if (index__ == 13) or (index__ == 14):
                    three.append((int(x), int(y)))
                  cv2.circle(src_img, (int(x), int(y)), 5, (0,0,255), -1)
                if (index__ == 0) or (index__ == 1) or (index__ == 2) or (index__ == 3) or (index__ == 4) or (index__ == 15) or (index__ == 16):
                  if (index__ == 15) or (index__ == 16):
                    # foot
                    four.append((int(x), int(y)))
                  else:
                    if (index__ == 0): 
                       # only use norse
                       head_y.append(int(y))
                  cv2.circle(src_img, (int(x), int(y)), 5, (0,0,255), -1)
                #cv2.circle(src_img, (int(x), int(y)), 5, (0,0,255), -1)
              start, mid, end = (one[0] + one[1]), (two[0] + two[1]), (three[0] + three[1])
              foot_mid = (four[0] + four[1])
              # print(start, mid, end)
              vec1 = np.array([((start[0] + start[2]) / 2) - ((mid[0] + mid[2]) / 2), ((start[1] + start[3]) / 2) - ((mid[1] + mid[3]) / 2)])
              vec2 = np.array([((end[0] + end[2]) / 2) - ((mid[0] + mid[2]) / 2), ((end[1] + end[3]) / 2) - ((mid[1] + mid[3]) / 2)])
              x, y = vec1, vec2
              Lx=np.sqrt(x.dot(x))
              Ly=np.sqrt(y.dot(y))
              cos_angle=x.dot(y)/(Lx*Ly)
              angle=np.arccos(cos_angle)
              angle=angle*360/2/np.pi
              if (w > asp * h) and (angle > 100):
                count += 1
                # print(w, h, angle)
              print("==============")
              print(head_y)
              foot_mid_x = int((foot_mid[0] + foot_mid[2]) / 2)
              foot_mid_y = int((foot_mid[1] + foot_mid[3]) / 2)
              print(foot_mid_y)
              cv2.circle(src_img, (int(foot_mid_x), int(foot_mid_x)), 5, (0,0,255), -1)

              cv2.line(src_img, (int((start[0] + start[2]) / 2), int((start[1] + start[3]) / 2)), 
                        (int((mid[0] + mid[2]) / 2), int((mid[1] + mid[3]) / 2)), 
                       (0, 255, 0), 5)
              cv2.line(src_img, (int((end[0] + end[2]) / 2), int((end[1] + end[3]) / 2)), 
                        (int((mid[0] + mid[2]) / 2), int((mid[1] + mid[3]) / 2)), 
                       (0, 255, 0), 5)
              
              # print(keypoints[idx])
      # fig = plt.figure(figsize=(10, 10))
      # src_img = Image.fromarray(cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)).convert("RGB") # cv2PIL
      # src_img = torch.from_numpy(src_img) # cv2torch
      # ax = fig.add_subplot(2, 1, 1)
      # ax.imshow(src_img)
      # masks_ = standard_transforms.ToPILImage()(masks_).convert("RGB") # torch2PIL 
      # ax = fig.add_subplot(2, 1, 2)
      # ax.imshow(masks_)
      # src_img = standard_transforms.ToPILImage()(src_img).convert("RGB") # torch2PIL      
      # plt.imshow(src_img)
      # mask = cv2.cvtColor(numpy.asarray(mask),cv2.COLOR_RGB2BGR)
      # img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
      src_img = src_img[:,:,::-1] 
      plt.imshow(src_img)
      plt.show()
if __name__ == "__main__":
    main()