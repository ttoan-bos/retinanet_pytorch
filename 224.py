import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

torch.set_printoptions(sci_mode=False)

def cosine_sim(a, b):
    return F.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()


def pcc(a, b):
    a = a.flatten()
    b = b.flatten()

    a = a - a.mean()
    b = b - b.mean()

    return (a @ b) / (torch.norm(a) * torch.norm(b))






def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list, raw_output=False):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(
        model_path,
        weights_only=False,
        map_location=torch.device("cpu")
    )


    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))

        if image is None:
            continue

        image_orig = image.copy()
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)

        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        print("Input shape: ", image_orig.shape)
        print("Resized shape: ", image.shape)
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()

            torch.manual_seed(42)
            # image = torch.empty(8,3,224,224).uniform_(-5,5)
            image = image.repeat(8,1,1,1)
            print(image)
            if (raw_output):
                cls, reg, _ = model(image.float(), return_raw=True)
                # print("CLS:")
                # print(cls)
                # print("REG:")
                # print(reg)

                payload = torch.load("debug_tensors.pt", map_location="cpu")

                cls_ref = payload["cls"]
                reg_ref = payload["reg"]

                print(cls_ref.shape, cls_ref.dtype)
                print(reg_ref.shape, reg_ref.dtype)

                print("CLS cosine:", cosine_sim(cls_ref, cls))
                print("REG cosine:", cosine_sim(reg_ref, reg))

                print("CLS PCC:", pcc(cls_ref, cls).item())
                print("REG PCC:", pcc(reg_ref, reg).item())

                return 

            scores, classification, transformed_anchors = model(image.float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            h_orig, w_orig = image_orig.shape[:2]
            scale_x = w_orig / 224.0
            scale_y = h_orig / 224.0

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)

                label_name = labels[int(classification[idxs[0][j]])]
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.imshow('detections', image_orig)
            cv2.waitKey(0)
            # break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')
    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument('--raw_output', action='store_true', help='If true, model outputs raw tensors')
    parser = parser.parse_args()
    detect_image(parser.image_dir, parser.model_path, parser.class_list, parser.raw_output)

