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
Image.MAX_IMAGE_PIXELS = 933120000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# no mean_std 0
# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet mean and std 
# mean_std = ([0.49331234, 0.47941793, 0.4508255 ], [0.28019103, 0.27430675, 0.28256588]) # testA 
# mean_std = ([0.47262473, 0.4463122, 0.42317524], [0.28220754, 0.27441582, 0.27632682]) # testB4
# mean_std = ([0.4725966, 0.4463304, 0.4231564], [0.28216869, 0.27438522, 0.27628942]) # testB3
# mean_std = ([0.47256242, 0.44635145, 0.42313362], [0.28213677, 0.27434426, 0.27624931]) # testB2



img_transform = standard_transforms.Compose([
    # standard_transforms.Resize((1024, 768)),  # direct rescale image size to fixed size would cause performance drop
    standard_transforms.ToTensor(),
    # standard_transforms.Normalize(*mean_std)  # norm data with mean and std would cause performance drop
])

def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)

@torch.no_grad()
def main():
    image_path = './B/'  # image path
    model_name = 'keypointrcnn_resnet50_fpn'  # model name, choose from 'keypointrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn' 
    dataset = 'coco'  # model pretrain dataset
    score = 0.8 # objectness score threshold
    asp = 0.7  # length and width asptio threshold
    angle_th = 80  # the angle threshold between hip-shoulder vector and  hip-knee vector 
    
    num_classes = 2  # 2 for keypointrcnn_resnet50_fpn, 91 for maskrcnn_resnet50_fpn and fasterrcnn_resnet50_fpn
    names = coco_names.names
    bs = 2  # batch size

    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[model_name](num_classes=num_classes, pretrained=True)  # supported by the newest pytorch
    
    model = model.cuda()
    model.eval()

    import csv
    # 1. creat csv file
    f = open('./fall_result.csv', 'w', encoding='utf-8', newline='')
    # 2. creat csv wrire
    csv_writer = csv.writer(f)
    # 3. creat table head
    # csv_writer.writerow(["file", "gt", "fall_count"])
    csv_writer.writerow(["file", "fall_count"])
    image_count = 0
    image_list = os.listdir(image_path)
    # assert len(image_list) == 5775  # for A testset 
    # assert len(image_list) == 2641 # for B testset 
    for iter_count in range((len(image_list) // bs) + 1):
      if iter_count != (len(image_list) // bs):
        image_list_ = image_list[iter_count * bs : (iter_count + 1) * bs]
      else:
        image_list_ = image_list[iter_count * bs : ]

      input = []
      for index, image_name in enumerate(image_list_):      
        image_dir = os.path.join(image_path, image_name)
        # print(image_name)
        src_img = Image.open(image_dir)
        src_img = src_img.convert('RGB') 
        src_img = img_transform(src_img)
        # print(src_img.shape)
        # src_img = cv2.imread(image_dir)
        # print(image_name, src_img.shape)
        # img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
        img_tensor = src_img.cuda()
        input.append(img_tensor)
        
      # get the model output
      out = model(input)
      
      for index, image_name in enumerate(image_list_):
        boxes = out[index]['boxes']
        labels = out[index]['labels']
        scores = out[index]['scores']
        # masks = out[index]['masks']
        keypoints = out[index]['keypoints']

        count = 0
        for idx in range(boxes.shape[0]):
            if scores[idx] >= score:
                # fliter the score which lower than 0.8
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]  # get each bounging box's 4 position
                # whcih are left top, left bottom, right top, and right bottom, respectly
                # print(names[str(labels[idx].item())], scores[idx])
                if names[str(labels[idx].item())] != 'person':
                  # fliter the class which not belong to person
                  continue
                w, h = x2 - x1, y2 - y1  # get the width and length of bounding box
                # print(w,h)
                one = []  # save the shoulders position
                two = []  # save the hips position
                three = [] # save the knees position
                four = [] # save the foot position
                head_y = []  # save the head position
                for index__, point in enumerate(keypoints[idx]):
                  x, y, vis = point
                  x, y, vis = x.item(), y.item(), vis.item()
                  if vis == 0:
                    # fliter the in-visible keypoint
                    continue
                  if (index__ == 5) or (index__ == 6) or (index__ == 11) or (index__ == 12) or (index__ == 13) or (index__ == 14):
                    # shoulder, hip or knee
                    if (index__ == 5) or (index__ == 6):
                      # two shoulders
                      one.append((int(x), int(y)))
                    if (index__ == 11) or (index__ == 12):
                      # two hips
                      two.append((int(x), int(y)))
                    if (index__ == 13) or (index__ == 14):
                      # two knees
                      three.append((int(x), int(y)))
                  
                  if (index__ == 0) or (index__ == 1) or (index__ == 2) or (index__ == 3) or (index__ == 4) or (index__ == 15) or (index__ == 16):
                    # five head keypoints and two foots
                    if (index__ == 15) or (index__ == 16):
                      # foots
                      four.append((int(x), int(y)))
                    else:
                      if (index__ == 0): 
                         # only use norse
                         head_y.append(int(y))
                start, mid, end = (one[0] + one[1]), (two[0] + two[1]), (three[0] + three[1])
                # print(start, mid, end)
                vec1 = np.array([((start[0]+start[2]) / 2) - ((mid[0]+mid[2]) / 2), ((start[1]+start[3]) / 2) - ((mid[1]+mid[3]) / 2)])  # hip-shoulder vector
                vec2 = np.array([((end[0] + end[2]) / 2) - ((mid[0] + mid[2]) / 2), ((end[1] + end[3]) / 2) - ((mid[1] + mid[3]) / 2)]) # hip-knee vector
                x, y = vec1, vec2
                Lx=np.sqrt(x.dot(x))
                Ly=np.sqrt(y.dot(y))
                cos_angle=x.dot(y)/(Lx*Ly) # compute the cosine value
                angle=np.arccos(cos_angle)
                angle=angle*360/2/np.pi # transfer to angle
                
                foot_mid = (four[0] + four[1]) 
                # get the middle point position of two foots
                foot_mid_x = int((foot_mid[0] + foot_mid[2]) / 2)
                foot_mid_y = int((foot_mid[1] + foot_mid[3]) / 2)
                # get the head y position
                head_y_pos = head_y[0]
                # get the foot y position
                foot_y_pos = foot_mid_y
                
                # judge fal down
                if (w > asp * h) and (angle > angle_th):
                  count += 1
                else:  
                  if head_y_pos >= foot_y_pos:
                    print(head_y_pos, foot_y_pos)  
                    count += 1           
                #   print(w, h, angle)
                
                if count >= 10:  # here the count can be replace the cround count which detected by C3-framework, we use this in test A but no time in test B
                    # if too many people, we think they are must be spectators
                    count = 1
                name = names.get(str(labels[idx].item()))
                # cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
                # cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))
        # 4. write csv file
        try:
            csv_writer.writerow([image_name, count])
            image_count += 1
            print('{} people fall down in {}, {}-th image finished ...'.format(count, image_name, image_count))
            torch.cuda.empty_cache()
        except Exception:
            print("Image {} can not save to csv file...".format(image_name))
        # csv_writer.writerow([filename, gt, pred])
    # 5. close
    f.close()
    # from  matplotlib import pyplot as plt
    # %matplotlib inline
    # src_img = src_img[:,:,::-1] 
    # plt.imshow(src_img)
    # plt.show()
    # the following cv2.imshow is disabled in colab/jupyter
    # cv2.imshow('result', src_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)