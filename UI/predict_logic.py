import cv2
import numpy as np
from skimage import feature
from joblib import load  # Import joblib to load model and scaler

# Global thresholds
conf_threshold = 0.95  # Minimum confidence for valid detections
iou_threshold = 0.1  # IoU threshold for Non-Maximum Suppression (NMS)

# Load model and scaler
model = load(r"C:\GitHub\ML project\SVM-TrafficSigns\UI\svm_model.pkl")
scaler = load(r"C:\GitHub\ML project\SVM-TrafficSigns\UI\scaler.pkl")
label_encoder = load(r"C:\GitHub\ML project\SVM-TrafficSigns\UI\label_encoder.pkl")

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    hog_feature = feature.hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2",
    )
    return hog_feature

def sliding_window(img, window_sizes, stride):
    img_height, img_width = img.shape[:2]
    windows = []
    for window_size in window_sizes:
        window_width, window_height = window_size
        for ymin in range(0, img_height - window_height + 1, stride):
            for xmin in range(0, img_width - window_width + 1, stride):
                xmax = xmin + window_width
                ymax = ymin + window_height
                windows.append([xmin, ymin, xmax, ymax])
    return windows

def pyramid(img, scale=0.8, min_size=(30, 30)):
    acc_scale = 1.0
    pyramid_imgs = [(img, acc_scale)]
    while True:
        acc_scale *= scale
        w = int(img.shape[1] * acc_scale)
        h = int(img.shape[0] * acc_scale)
        if h < min_size[1] or w < min_size[0]:
            break
        img = cv2.resize(img, (w, h))
        pyramid_imgs.append((img, acc_scale))
    return pyramid_imgs


def visualize_bbox(img, bboxes, label_encoder):
    """
    Draw bounding boxes and labels on the image.

    Args:
        img (numpy.ndarray): The input image.
        bboxes (list): List of bounding boxes in the format [xmin, ymin, xmax, ymax, predict_id, confidence].
        label_encoder: The label encoder used to decode class labels.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for visualization

    for box in bboxes:
        xmin, ymin, xmax, ymax, predict_id, conf_score = box

        # Draw the bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Get the class name from the label encoder
        classname = label_encoder.inverse_transform([predict_id])[0]
        label = f"{classname} {conf_score:.2f}"

        # Add label above the bounding box
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + label_width, ymin), (0, 255, 0), -1)  # Label background
        cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Optionally, return the image with drawn bounding boxes (can be used for saving or further processing)
    return img


def compute_iou(bbox, bboxes, bbox_area, bboxes_area):
    xxmin = np.maximum(bbox[0], bboxes[:, 0])
    yymin = np.maximum(bbox[1], bboxes[:, 1])
    xxmax = np.minimum(bbox[2], bboxes[:, 2])
    yymax = np.minimum(bbox[3], bboxes[:, 3])

    w = np.maximum(0, xxmax - xxmin + 1)
    h = np.maximum(0, yymax - yymin + 1)

    intersection = w * h
    iou = intersection / (bbox_area + bboxes_area - intersection)
    return iou

def nms(bboxes, iou_threshold):
    if not bboxes:
        return []
    scores = np.array([bbox[5] for bbox in bboxes])
    sorted_indices = np.argsort(scores)[::-1]
    xmin = np.array([bbox[0] for bbox in bboxes])
    ymin = np.array([bbox[1] for bbox in bboxes])
    xmax = np.array([bbox[2] for bbox in bboxes])
    ymax = np.array([bbox[3] for bbox in bboxes])
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)
    keep = []
    while sorted_indices.size > 0:
        i = sorted_indices[0]
        keep.append(i)
        iou = compute_iou(
            [xmin[i], ymin[i], xmax[i], ymax[i]],
            np.array([xmin[sorted_indices[1:]], ymin[sorted_indices[1:]], xmax[sorted_indices[1:]],
                      ymax[sorted_indices[1:]]]).T,
            areas[i],
            areas[sorted_indices[1:]]
        )
        idx_to_keep = np.where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[idx_to_keep + 1]
    return [bboxes[i] for i in keep]

def predict(img, label_encoder):
    pyramid_imgs = pyramid(img)
    bboxes = []
    for pyramid_img, scale_factor in pyramid_imgs:
        window_list = sliding_window(pyramid_img, [(32, 32), (64, 64), (128, 128)], stride=12)
        for window in window_list:
            xmin, ymin, xmax, ymax = window
            object_img = pyramid_img[ymin:ymax, xmin:xmax]
            preprocessed_img = preprocess_img(object_img)
            normalized_img = scaler.transform([preprocessed_img])
            decision = model.predict_proba(normalized_img)[0]

            # Apply confidence threshold
            if np.all(decision < conf_threshold):
                continue

            predict_id = np.argmax(decision)
            conf_score = decision[predict_id]
            orig_xmin = int(xmin / scale_factor)
            orig_ymin = int(ymin / scale_factor)
            orig_xmax = int(xmax / scale_factor)
            orig_ymax = int(ymax / scale_factor)
            bboxes.append([orig_xmin, orig_ymin, orig_xmax, orig_ymax, predict_id, conf_score])

    # Apply Non-Maximum Suppression
    filtered_bboxes = nms(bboxes, iou_threshold)

    # Visualize bounding boxes
    annotated_img = visualize_bbox(img, filtered_bboxes, label_encoder)

    return annotated_img, filtered_bboxes


