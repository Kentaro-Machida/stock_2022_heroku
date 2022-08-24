"""
ストックの画像を受け取り、
bbox_searcherとpytorchを使用し、
ストックの位置と八重率を特定。
"""
from cv2 import COLOR_BGR2RGB
import cv2
import numpy as np
import torch
import configparser
#import torchvision.models as model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from bbox_searcher import Bbox_Getter

def get_RoIs(img:np.ndarray, boxes, out_size=224) -> list:
    """
    input
    image, list of bounding box
    output
    list of np.ndarray(W, H, C)
    """
    cut_img_list = []
    for x1, y1, x2, y2 in boxes:
        RoI = img[y1:y2, x1:x2]
        RoI = cv2.resize(RoI, (out_size, out_size))
        cut_img_list.append(RoI)

    return cut_img_list

def tensor_preprocess(img:np.ndarray)->torch.Tensor:
    """"
    画像の前処理を行う.
    画素値範囲を[0, 1] and (W, H, C)を(C, W, H)に変換
    image netの平均画素と分散で標準化
    """
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensorer = transforms.ToTensor()
    input = tensorer(img)
    input = normalizer(input)
    input = torch.unsqueeze(input, 0)
    return input

def get_detected_image(config_path ,img_path)->tuple:
    """
    load image and detects stock.
    output (bool, np.ndarray)
    (検出されたかどうか, 画像)
    """

    # get Bounding Boxes with parameters in "roi_config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)

    img_size=(int(config['DEFAULT']['img_size']), int(config['DEFAULT']['img_size']))
    img = cv2.imread(img_path)
    bgr_img = cv2.resize(img, img_size)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    masking_type = config['DEFAULT']['masking_type']

    b_thresh = (int(config['DEFAULT']['b_thresh_l']),
        int(config['DEFAULT']['b_thresh_h']))

    g_thresh = (int(config['DEFAULT']['g_thresh_l']),
        int(config['DEFAULT']['g_thresh_h']))

    r_thresh = (int(config['DEFAULT']['r_thresh_l']),
        int(config['DEFAULT']['r_thresh_h']))

    h_thresh = (int(config['DEFAULT']['h_thresh_l']),
    int(config['DEFAULT']['h_thresh_h']))

    s_thresh = (int(config['DEFAULT']['s_thresh_l']),
        int(config['DEFAULT']['s_thresh_h']))

    v_thresh = (int(config['DEFAULT']['v_thresh_l']),
        int(config['DEFAULT']['v_thresh_h']))

    area_low_thresh_rate=1/float(config['DEFAULT']['num_per_height_h'])
    area_high_thresh_rate=1/float(config['DEFAULT']['num_per_height_l'])

    aspect_low_thresh=float(config['DEFAULT']['aspect_low_thresh'])
    aspect_high_thresh=float(config['DEFAULT']['aspect_high_thresh'])

    closing_ksize = (int(config['DEFAULT']['closing_ksize']), int(config['DEFAULT']['closing_ksize']))
    opening_ksize = (int(config['DEFAULT']['opening_ksize']), int(config['DEFAULT']['opening_ksize']))

    getter = Bbox_Getter(
        b_thresh, g_thresh, r_thresh,
        area_low_thresh_rate, area_high_thresh_rate,
        masking_type,
        aspect_low_thresh, aspect_high_thresh,
        closing_ksize, opening_ksize,
        h_thresh, s_thresh, v_thresh
        )

    boxes = getter.get_bbox(bgr_img)
    if len(boxes) == 0:
        return (False, bgr_img)  # エラーのリターン

    # RoI画像を抽出
    RoIs = get_RoIs(rgb_img, boxes)

    # すべてのRoIに前処理関数をかます
    RoIs = list(map(tensor_preprocess, RoIs))

    # すべてのRoI4次元tensorを連結
    input_batch = torch.cat(RoIs)

    trained_model = torch.load(config['DEFAULT']['model_path'])

    # 確率の計算
    logits = trained_model(input_batch)
    m = torch.nn.Softmax(dim=1)
    outs = m(logits)
    
    count = 0
    for x1, y1, x2, y2 in boxes:
        prob = float(outs[count][0])
        color = (255*prob,255- 255*prob, 255)
        out_img = cv2.rectangle(bgr_img, (x1, y1),(x2, y2),color=color, thickness=4)
        out_img = cv2.putText(bgr_img, str(int(prob*100)), org=(x1, y1),
        color=color, fontFace=1,fontScale=2, thickness=2)
        count += 1

    return (True, out_img)