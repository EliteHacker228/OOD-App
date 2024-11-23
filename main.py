import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np

model = YOLO('model/obb_model_s_yolo.pt')

def detect_objects(image_path, output_path):
    img = cv2.imread(image_path)
    results = model(img)
    annotated_img = results[0].plot()  # Use .plot() instead of .render()

    for result in results:
        if result.obb is not None and len(result.obb) != 0:
            for obb in result.obb:
                xywhr = obb.xywhr[0]
                x_center, y_center, width, height, rotation = xywhr
                angle_degrees = rotation.item() * (180 / np.pi)

                print(f"Angle: {angle_degrees:.2f} degrees")
        else:
            print("No oriented bounding boxes detected.")

    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    # plt.show()
    cv2.imwrite(output_path, annotated_img)

for name in ['Рисунок (61).png',
             'Рисунок (208).png',
             'Рисунок (214).png',
             'Рисунок (215).png',
             'Рисунок (217).png',
             'Рисунок (315).png']:

    image_name= name
    image_path = f"imgs/{image_name}"
    output_path = f"results/{image_name}"
    detect_objects(image_path, output_path)