import matplotlib.pyplot as plt
import cv2

from super_gradients.training import models
from super_gradients.common.object_names import Models

from object_detection.constants import CLASS_NAMES
from object_detection.utils import get_class_colors, get_prediction_from_model, plot_prediction


if __name__ == '__main__':
    img = cv2.imread('../images/table_chairs1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    yolo_model = models.get(Models.YOLO_NAS_S, pretrained_weights='coco').cuda()
    
    class_colors = get_class_colors(CLASS_NAMES)
    bboxes, class_names, class_indices, confidences = get_prediction_from_model(img, yolo_model)
    img = plot_prediction(img, class_names, bboxes, class_indices, confidences, class_colors)
        
    plt.imsave('../images/table_chairs1_res.jpg', img)
    
    