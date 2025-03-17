import random
import cv2


def plot_prediction(img, class_names, bboxes, class_indices, confidences, class_colors):
    for i in range(len(bboxes)):
            x_min, y_min, x_max, y_max = map(int, bboxes[i])
            class_name = class_names[class_indices[i]]
            label = f"{class_name}: {confidences[i]:.2f}"

            color = class_colors[class_name]

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
            text_x, text_y = x_min, max(y_min - 5, text_height + 5)
            text_bg_x_max, text_bg_y_max = text_x + text_width + 6, text_y - text_height - 6

            cv2.rectangle(img, (text_x, text_y - text_height - 6), (text_bg_x_max, text_y), color, -1)

            cv2.putText(img, label, (text_x + 3, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def get_class_colors(class_names):
    """
    Generate random colors for each class
    """
    random.seed(42)
    return {class_name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for class_name in class_names}


def get_prediction_from_model(img, model):
    """
    
    """
    prediction = model.predict(img, fuse_model=False)
    predictions_dict = prediction.prediction
    
    bboxes = predictions_dict.bboxes_xyxy
    class_names = prediction.class_names
    class_indices = predictions_dict.labels.astype(int) 
    confidences =  predictions_dict.confidence.astype(float)

    return bboxes, class_names, class_indices, confidences

