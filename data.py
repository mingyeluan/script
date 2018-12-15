import json
import cv2
import mxnet as mx
import subprocess
import os

json_path = 'C:/Users/mingye.luan/Desktop/day8/carnet_realtest_20181203_v20181213_2/carnet_realtest_20181203_v20181213_2/DMS_RAW_Nebula_20100101-080045_567.json'
images_path = 'C:/Users/mingye.luan/Desktop/day8/DATA/carnet_realtest_20181203_test_hand/carnet_realtest_20181203/DMS_RAW_Nebula_20100101-080045_567/*.png'
keypoint_39 = [20]

name_list = [images_path + f for f in os.listdir(images_path) if f.endswith('png')]

with open('train.lst', 'w') as f:
    count = 0
    json_file = open(json_path, 'r').readlines()
    for line in json_file:
        json_dict = json.loads(line)
        if json_dict['head'][0]['attrs']['ignore'] == 'no':
            key = json_dict['face_keypoint_39'][0]['data']
            image_name = json_dict['image_key']

            face_data = json_dict['head'][0]['data']
            length = int(max(face_data[2]-face_data[0],face_data[3]-face_data[1]))

            x1 = int(face_data[0])
            y1 = int(face_data[1])
            x2 = int(face_data[2])
            y2 = int(face_data[3])

            center = [int((x2+x1)/2),int((y2+y1)/2)]

            img = cv2.imread(images_path + image_name, 1)

            x1 = int(center[0] - length / 2)
            y1 = int(center[1] - length / 2)
            x2 = int(center[0] + length / 2)
            y2 = int(center[1] + length / 2)

            crop_img = img[y1:y2,x1:x2]
            crop_img = cv2.resize(crop_img,(64,64))
            if not os.path.exists('D:/program/graduation design/coding/data/temp/resized/'):
              os.mkdir('D:/program/graduation design/coding/data/temp/resized/')
            cv2.imwrite('D:/program/graduation design/coding/data/temp/resized/'+image_name,crop_img)

            resize_path='D:/program/graduation design/coding/data/temp/resized/'

            write_line = ''
            for index in keypoint_39:
                write_line += str(int((64*(key[int(index)][0]-x1))/(x2-x1)))
                write_line += '\t'
                write_line += str(int((64*(key[int(index)][1]-y1))/(y2-y1)))
                write_line += '\t'

            f.write(str(count) + '\t' + write_line + resize_path + image_name + '\n')
            count += 1

'''
            img_plot =cv2.imread(images_path+image_name,1)
            for i in keypoint_39:
                cv2.circle(img_plot,(int(key[int(i)][0]),int(key[int(i)][1])),4,(0,0,255),-1)
            if not os.path.exists('D:/program/graduation design/coding/data/temp/examples/'):
              os.mkdir('D:/program/graduation design/coding/data/temp/examples/')
            cv2.imwrite('D:/program/graduation design/coding/data/temp/examples/'+image_name,img_plot)
            #print(crop_img.shape,img_plot.shape)


            img_plot = cv2.imread(resize_path + image_name, 1)
            for i in keypoint_39:
                cv2.circle(img_plot, (int((64*(key[int(i)][0]-x1))/(x2-x1)), int((64*(key[int(i)][1]-y1))/(y2-y1))), 1, (0, 0, 255), -1)
            if not os.path.exists('D:/program/graduation design/coding/data/temp/new/'):
                os.mkdir('D:/program/graduation design/coding/data/temp/new/')
            cv2.imwrite('D:/program/graduation design/coding/data/temp/new/' + image_name, img_plot)
'''

im2rec_path=os.path.join(mx.__path__[0],'tools/im2rec.py')
if not os.path.exists(im2rec_path):
    im2rec_path=os.path.join(os.path.dirname(os.path.dirname(mx.__path__[0])),'tools/im2rec.py')
subprocess.check_call(['python',im2rec_path,os.path.abspath('train.lst'),os.path.abspath('./'),'--pack-label'])


