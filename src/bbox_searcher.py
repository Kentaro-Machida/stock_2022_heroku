import cv2
import numpy as np
import sys
from dataclasses import dataclass

@dataclass
class Bbox_Getter:
    b_thresh: tuple
    g_thresh: tuple
    r_thresh: tuple

    area_low_thresh: int
    area_high_thresh: int

    masking_type: str = "rgb" #  rgb or hsv

    aspect_low_thresh: float = 0.7
    aspect_high_thresh: float = 1.3

    closing_ksize: tuple = (5, 5)
    opening_ksize: tuple = (10, 10)

    h_thresh: tuple = (30, 90)
    s_thresh: tuple = (64, 255)
    v_thresh: tuple = (0, 255)


    def get_binary_image(self, img:np.ndarray,
                        b_thresh=(0, 255), g_thresh=(0, 255), r_thresh=(0, 255))->np.ndarray:
        '''
        Outputs a binarized image according to BGR conditions.
        Please input image loaded by 'cv2.imread()'.
        input: np.ndarray(B, G, R)
        output: binary image
        '''
        thresh_list = [b_thresh, g_thresh, r_thresh]
        binary_list = []

        for i, thresh in enumerate(thresh_list):
            binary_img = np.uint8((img[:,:,i] > thresh[0])*(img[:,:,i]<thresh[1]))
            binary_list.append(binary_img)

        final_binary = binary_list[0] * binary_list[1] * binary_list[2]
        return final_binary


    def get_binary_image_hsv(self, img:np.ndarray,
                        h_thresh=(30, 90), s_thresh=(64, 255), v_thresh=(0, 255))->np.ndarray:
        '''
        Outputs a binarized image according to HSV conditions.
        Please input image loaded by 'cv2.imread()'.
        input: np.ndarray(B, G, R)
        output: binary image
        '''
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2HSV)

        thresh_list = [h_thresh, s_thresh, v_thresh]
        binary_list = []

        for i, thresh in enumerate(thresh_list):
            binary_img = np.uint8((img[:,:,i] > thresh[0])*(img[:,:,i]<thresh[1]))
            binary_list.append(binary_img)

        final_binary = binary_list[0] * binary_list[1] * binary_list[2]
        return final_binary


    def closing(self, img:np.ndarray)->np.ndarray:
        '''
        Eliminate noise by closing process.
        input: binary image
        output: closed image
        '''
        kernel = np.ones(self.closing_ksize, np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        return erosion


    def opening(self, img:np.ndarray)->np.ndarray:
        '''
        Reattach areas that have been separated by closing.
        input: binary image
        output: opened image
        '''
        kernel = np.ones(self.opening_ksize, np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)
        return dilation


    def get_bbox_from_labeled_binary(self, binary_img:np.ndarray)->list:
        '''
        Extract the area from the binary image and get Bounding Box.
        input: binary image
        output: list of bbox tuple(x1, y1, x2, y2)
        '''
        boxes = []
        label_num, labels = cv2.connectedComponents(binary_img)
        for target in range(label_num):
            mask = (labels == target)
            index_list = mask.nonzero()
            y1 = np.min(index_list[0])
            y2 = np.max(index_list[0])
            x1 = np.min(index_list[1])
            x2 = np.max(index_list[1])
            boxes.append((x1, y1, x2, y2))
        return boxes


    def describe_bbox(self, img:np.ndarray, boxes:list)->None:
        '''
        Show image with Bbox.
        '''
        window_name = "stock with Bounding Box"
        for box in boxes:
            start = (box[0], box[1])
            end = (box[2], box[3])
            img = cv2.rectangle(img, start, end, color=(0, 0, 255), thickness=2)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)


    def choice_by_area(self, boxes:list, low_thresh_rate:float, high_thresh_rate:float,
                        img_size:tuple)->list:
        '''
        remove images by area value.
        caluculate mean and remove outliers.
        '''
        area_list = []
        final_list = []
        print(f"befor area choice: {len(boxes)}")
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width*height
            area_list.append(area)

        low_thresh = img_size[0]*img_size[1]*(low_thresh_rate**2)
        high_thresh = img_size[0]*img_size[1]*(high_thresh_rate**2)

        for box, area in zip(boxes, area_list):
            if(area > low_thresh)&(area < high_thresh):
                final_list.append(box)

        print(f"after area choice: {len(final_list)}")
        return final_list


    def choice_by_aspect(self, boxes:list, low_thresh:int, high_thresh:int):
        '''
        remove images by aspect ratio.
        '''
        print(f"befor aspect choice: {len(boxes)}")
        final_list = []
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            aspect = height/width
            if (aspect > low_thresh) & (aspect < high_thresh):
                final_list.append(box)
        print(f"after aspect choice: {len(final_list)}")
        return final_list


    def describe_binary(self, img:np.ndarray)->None:
        '''
        show binary image
        '''
        
        if self.masking_type=='rgb':
            binary = self.get_binary_image(img, b_thresh=self.b_thresh,
                                         g_thresh=self.g_thresh, r_thresh=self.r_thresh)
            window_name = "binary image RGB thresh: ({}, {}, {})".format(
                self.b_thresh,self.g_thresh,self.r_thresh)
        elif self.masking_type=='hsv':
            binary = self.get_binary_image_hsv(img, h_thresh=self.h_thresh,
                                         s_thresh=self.s_thresh, v_thresh=self.v_thresh)
            window_name = "binary image HSV thresh: ({}, {}, {})".format(
                self.h_thresh,self.s_thresh,self.v_thresh)

        cv2.imshow(window_name, binary*255)
        cv2.waitKey(0)


    def describe_closed(self, img:np.ndarray)->None:
        '''
        show closed image
        '''
        if self.masking_type=='rgb':
            binary = self.get_binary_image(img, b_thresh=self.b_thresh,
                                            g_thresh=self.g_thresh, r_thresh=self.r_thresh)
        elif self.masking_type=='hsv':
            binary = self.get_binary_image_hsv(img, h_thresh=self.h_thresh,
                                         s_thresh=self.s_thresh, v_thresh=self.v_thresh)
        closed = self.closing(binary)
        window_name = "closing image {} thresh: ({}, {}, {})".format(
            self.masking_type,self.b_thresh,self.g_thresh,self.r_thresh)
        window_name += " kernel_size: {}".format(self.closing_ksize)

        cv2.imshow(window_name, closed*255)
        cv2.waitKey(0)


    def describe_opened(self, img:np.ndarray)->None:
        '''
        show opened image
        '''
        if self.masking_type=='rgb':
            binary = self.get_binary_image(img, b_thresh=self.b_thresh,
                                            g_thresh=self.g_thresh, r_thresh=self.r_thresh)
        elif self.masking_type=='hsv':
            binary = self.get_binary_image_hsv(img, h_thresh=self.h_thresh,
                                         s_thresh=self.s_thresh, v_thresh=self.v_thresh)
        closed = self.closing(binary)
        opened = self.opening(closed)

        window_name = "opened image {} thresh: ({}, {}, {})".format(
            self.masking_type,self.b_thresh,self.g_thresh,self.r_thresh)
        window_name += " closing_k_size: {}".format(self.closing_ksize)
        window_name += " opening_k_size: {}".format(self.opening_ksize)

        cv2.imshow(window_name, opened*255)
        cv2.waitKey(0)


    def get_bbox(self, img:np.ndarray)->list:
        '''
        get Bounding box.
        input: image
        output: list of Bounding box
        '''
        img_shape = img.shape
        img_size = (img_shape[0], img_shape[1])

        if self.masking_type=="hsv":
            binary = self.get_binary_image_hsv(img, h_thresh=self.h_thresh,
                                         s_thresh=self.s_thresh, v_thresh=self.v_thresh)
        elif self.masking_type=='rgb':
            binary = self.get_binary_image(img, b_thresh=self.b_thresh,
                                            g_thresh=self.g_thresh, r_thresh=self.r_thresh)
        else:
            print("masking_type error.")
            sys.exit(1)

        closed = self.closing(binary)
        opened = self.opening(closed)
        boxes = self.get_bbox_from_labeled_binary(opened)
        filtered_by_area = self.choice_by_area(boxes,
            self.area_low_thresh, self.area_high_thresh, img_size)
        filtered_by_aspect = self.choice_by_aspect(filtered_by_area,
            self.aspect_low_thresh, self.aspect_high_thresh)

        return filtered_by_aspect

##### test
def main():
    masking_type = "hsv"
    img_size=(448,448)

    b_thresh = (0, 255)
    g_thresh = (128, 255)
    r_thresh = (0, 255)

    h_thresh = (110, 140)
    s_thresh = (180, 255)
    v_thresh = (150, 255)

    area_low_thresh_rate = 0.2
    area_high_thresh_rate = 9.5

    img_path = './data/sample_images/test_img.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)

    if masking_type=="rgb":
        getter = Bbox_Getter(b_thresh, g_thresh, r_thresh,
            area_low_thresh_rate, area_high_thresh_rate, masking_type=masking_type)
        boxes = getter.get_bbox(img)
    
    elif masking_type=='hsv':
        getter = Bbox_Getter(h_thresh, s_thresh, v_thresh,
            area_low_thresh_rate, area_high_thresh_rate, masking_type=masking_type)
        boxes = getter.get_bbox(img)

    # テスト どれか一つだけ実行
    # getter.describe_binary(img)
    getter.describe_bbox(img, boxes)
    # getter.describe_closed(img)
    # getter.describe_opened(img)

if __name__ == '__main__':
    main()