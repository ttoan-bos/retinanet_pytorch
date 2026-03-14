# import torch
# import numpy as np
# import time
# import os
# import csv
# import cv2
# import argparse
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def load_classes(csv_reader):
#     result = {}

#     for line, row in enumerate(csv_reader):
#         line += 1

#         try:
#             class_name, class_id = row
#         except ValueError:
#             raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
#         class_id = int(class_id)

#         if class_name in result:
#             raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
#         result[class_name] = class_id
#     return result


# # Draws a caption above the box in an image
# def draw_caption(image, box, caption):
#     b = np.array(box).astype(int)
#     cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
#     cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


# def detect_image(image_path, model_path, class_list):

#     with open(class_list, 'r') as f:
#         classes = load_classes(csv.reader(f, delimiter=','))

#     labels = {}
#     for key, value in classes.items():
#         labels[value] = key

#     model = torch.load(
#         model_path,
#         weights_only=False,
#         map_location=torch.device("cpu")
#     )


#     if isinstance(model, torch.nn.DataParallel):
#         model = model.module

#     if torch.cuda.is_available():
#         model = model.cuda()

#     model.training = False
#     model.eval()

#     for img_name in os.listdir(image_path):

#         image = cv2.imread(os.path.join(image_path, img_name))
#         if image is None:
#             continue
#         #image_orig = cv2.resize(image.copy(), (224, 224))
#         image_orig = image.copy()


#         rows, cols, cns = image.shape

#         smallest_side = min(rows, cols)

#         # rescale the image so the smallest side is min_side
#         min_side = 608
#         max_side = 1024
#         scale = min_side / smallest_side

#         # check if the largest side is now greater than max_side, which can happen
#         # when images have a large aspect ratio
#         largest_side = max(rows, cols)

#         if largest_side * scale > max_side:
#             scale = max_side / largest_side

#         # resize the image with the computed scale
#         image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
#         rows, cols, cns = image.shape

#         pad_w = 32 - rows % 32
#         pad_h = 32 - cols % 32

#         new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
#         new_image[:rows, :cols, :] = image.astype(np.float32)
#         image = new_image.astype(np.float32)


#          # FORCE FIXED INPUT SIZE
#         # image = cv2.resize(image, (224, 224))
#         # image = image.astype(np.float32)
#         # scale = 1.0
   

#         image /= 255
#         image -= [0.485, 0.456, 0.406]
#         image /= [0.229, 0.224, 0.225]
#         image = np.expand_dims(image, 0)
#         image = np.transpose(image, (0, 3, 1, 2))

#         with torch.no_grad():

#             image = torch.from_numpy(image)
#             if torch.cuda.is_available():
#                 image = image.cuda()

#             st = time.time()
#             print(image.shape, image_orig.shape, scale)

#             image = image.to(device)
#             print(image)
#             scores, classification, transformed_anchors = model(image.float())

#             print('Elapsed time: {}'.format(time.time() - st))
#             idxs = np.where(scores.cpu() > 0.5)
            
#             for j in range(idxs[0].shape[0]):
#                 bbox = transformed_anchors[idxs[0][j], :]

#                 x1 = int(bbox[0] / scale)
#                 y1 = int(bbox[1] / scale)
#                 x2 = int(bbox[2] / scale)
#                 y2 = int(bbox[3] / scale)
#                 label_name = labels[int(classification[idxs[0][j]])]
#                 print(bbox, classification.shape)
#                 score = scores[j]
#                 caption = '{} {:.3f}'.format(label_name, score)
#                 # draw_caption(img, (x1, y1, x2, y2), label_name)
#                 draw_caption(image_orig, (x1, y1, x2, y2), caption)
#                 cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

#             cv2.imshow('detections', image_orig)
#             cv2.waitKey(0)


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

#     parser.add_argument('--image_dir', help='Path to directory containing images')
#     parser.add_argument('--model_path', help='Path to model')
#     parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')

#     parser = parser.parse_args()

#     detect_image(parser.image_dir, parser.model_path, parser.class_list)

import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse

from retinanet.model import resnet50


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise ValueError(f'line {line}: format should be "class_name,class_id"')

        class_id = int(class_id)

        if class_name in result:
            raise ValueError(f'line {line}: duplicate class name: "{class_name}"')

        result[class_name] = class_id

    return result


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list):

    # Load classes
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {v: k for k, v in classes.items()}
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # Create model architecture
    model = resnet50(num_classes=num_classes, pretrained=False)

    # Load state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    for img_name in os.listdir(image_path):

        img_file = os.path.join(image_path, img_name)
        image = cv2.imread(img_file)

        if image is None:
            continue

        image_orig = image.copy()

        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)

        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        image = cv2.resize(
            image,
            (int(round(cols * scale)), int(round(rows * scale)))
        )

        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        image = new_image
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]

        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image).float().to(device)

            start = time.time()

            scores, classification, transformed_anchors = model(image)

            print("Elapsed time:", time.time() - start)

            scores = scores.cpu()
            classification = classification.cpu()
            transformed_anchors = transformed_anchors.cpu()

            idxs = np.where(scores > 0.5)

            for j in range(idxs[0].shape[0]):

                bbox = transformed_anchors[idxs[0][j]]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)

                label = int(classification[idxs[0][j]])
                label_name = labels[label]

                score = scores[idxs[0][j]]

                caption = f"{label_name} {float(score):.3f}"

                draw_caption(image_orig, (x1, y1, x2, y2), caption)

                cv2.rectangle(
                    image_orig,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

        cv2.imshow("detections", image_orig)
        cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simple script for visualizing RetinaNet detection results."
    )

    parser.add_argument("--image_dir", required=True,
                        help="Path to directory containing images")

    parser.add_argument("--model_path", required=True,
                        help="Path to state_dict model")

    parser.add_argument("--class_list", required=True,
                        help="CSV file containing class names")

    args = parser.parse_args()

    detect_image(args.image_dir, args.model_path, args.class_list)