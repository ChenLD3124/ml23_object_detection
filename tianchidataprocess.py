import numpy as np # linear algebra
import os
import json
from tqdm.auto import tqdm
import shutil as sh
import cv2

josn_path = "./partA/Annotations/anno_train.json"
image_path = "./partA/defect_Images/"

name_list = []
image_h_list = []
image_w_list = []
c_list = []
w_list = []
h_list = []
x_center_list = []
y_center_list = []

with open(josn_path, 'r') as f:
    temps = tqdm(json.loads(f.read()))
    for temp in temps:
        # image_w = temp["image_width"]
        # image_h = temp["image_height"]
        name = temp["name"].split('.')[0]
        path = os.path.join(image_path, temp["name"])
        # print('path: ',path)
        im = cv2.imread(path)
        sp = im.shape
        image_h, image_w = sp[0], sp[1]
        # print("image_h, image_w: ", image_h, image_w)
        # print("defect_name: ",temp["defect_name"])
        #bboxs
        x_l, y_l, x_r, y_r = temp["bbox"]
        # print(temp["name"], temp["bbox"])
        # 根据缺陷名称获取类别编号的函数
        def get_defect_category(defect_name):
            if defect_name in ["浆斑", "油渍","污渍","水渍"]:
                return '0'
            elif defect_name in ["三丝"]:
                return '1'
            elif defect_name in ["断氨纶"]:
                return '2'
            elif defect_name in ["毛粒", "烧毛痕"]:
                return '3'
            elif defect_name in ["结头"]:
                return '4'
            elif defect_name in ["修痕", "磨痕"]:
                return '5'
            elif defect_name in [ "破洞"]:
                return '6'
            elif defect_name in ["纬缩", "轧痕", "死皱"]:
                return '7'
            elif defect_name in ["粗维", "松经", "粗经", "筘路", "纬纱不良", "吊经"]:
                return '8'
            elif defect_name in ["浪纹档", "稀密档","色差档"]:
                return '9'
            elif defect_name in ["断经"]:
                return '10'
            elif defect_name in ["双纬", "云织", "双经", "整经结"]:
                return '11'
            elif defect_name in [ "花板跳", "百脚", "星跳", "跳花", "跳纱"]:
                return '12'
            else:
                print(defect_name)
                assert 1==0
                return '其他'
        defect_name=get_defect_category(temp["defect_name"])

        # print(image_w, image_h)
        # print(defect_name)
        x_center = (x_l + x_r)/(2*image_w)
        y_center = (y_l + y_r)/(2*image_h)
        w = (x_r - x_l)/(image_w)
        h = (y_r - y_l)/(image_h)
        # print(x_center, y_center, w, h)
        name_list.append(temp["name"])
        c_list.append(defect_name)
        image_h_list.append(image_w)
        image_w_list.append(image_h)
        x_center_list.append(x_center)
        y_center_list.append(y_center)
        w_list.append(w)
        h_list.append(h)

    index = list(set(name_list))
    print(len(index))
    for fold in [0]:
        val_index = index[len(index) * fold // 5:len(index) * (fold + 1) // 5]
        print(len(val_index))
        for num, name in enumerate(name_list):
            print(c_list[num], x_center_list[num], y_center_list[num], w_list[num], h_list[num])
            row = [c_list[num], x_center_list[num], y_center_list[num], w_list[num], h_list[num]]
            if name in val_index:
                path2save = 'val/'
            else:
                path2save = 'train/'
            # print('convertor\\fold{}\\labels\\'.format(fold) + path2save)
            # print('convertor\\fold{}/labels\\'.format(fold) + path2save + name.split('.')[0] + ".txt")
            # print("{}/{}".format(image_path, name))
            # print('convertor\\fold{}\\images\\{}\\{}'.format(fold, path2save, name))
            if not os.path.exists('convertor/fold{}/labels/'.format(fold) + path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold) + path2save)
            with open('convertor/fold{}/labels/'.format(fold) + path2save + name.split('.')[0] + ".txt", 'a+') as f:
                for data in row:
                    f.write('{} '.format(data))
                f.write('\n')
                if not os.path.exists('convertor/fold{}/images/{}'.format(fold, path2save)):
                    os.makedirs('convertor/fold{}/images/{}'.format(fold, path2save))
                sh.copy(os.path.join(image_path, name),
                        'convertor/fold{}/images/{}/{}'.format(fold, path2save, name))

