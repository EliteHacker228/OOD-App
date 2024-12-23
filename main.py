import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import os


def get_angle_from_result(obb):
    xywhr = obb.xywhr[0]
    x_center, y_center, width, height, rotation = xywhr
    angle_degrees = rotation.item() * (180 / np.pi)
    return round(angle_degrees, 2)


def detect_objects(model, image_path, output_path):
    img = cv2.imread(image_path)
    results = model(img)
    annotated_img = results[0].plot()

    for result in results:
        if result.obb is not None and len(result.obb) != 0:
            angle_sum = 0
            for obb in result.obb:
                angle_sum += get_angle_from_result(obb)
            res_angle = angle_sum / len(result.obb)
            print(f"Angle: {res_angle:.2f} degrees")
        else:
            print("No oriented bounding boxes detected.")

    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    cv2.imwrite(output_path, annotated_img)


model = YOLO('model/pubtab_1548_aug.pt')
for image_name in os.listdir('./imgs'):
    input_path = f"imgs/{image_name}"
    output_path = f"results/{image_name}"
    detect_objects(model, input_path, output_path)
