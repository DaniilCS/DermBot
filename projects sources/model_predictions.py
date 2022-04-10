import torchvision.models
import torch
import os
import time
import threading
import cv2
import numpy as np
from process_predictions import make_text_prediction, make_text_state_prediction, detection_prediction_text
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, Normalize, Resize, RandomRotation, RandomHorizontalFlip, ToTensor
from PIL import Image

def predict_disease(picture_path, user_id):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(512, 7)
    model.to("cpu")
    model.load_state_dict(torch.load(r"C:\Users\Даниил\Desktop\КТ1\pythonProject1\Models\AcneClassifier.pth", map_location=torch.device('cpu')))
    model.eval()
    img = Image.open(picture_path)
    transform = Compose(
        [
            Resize((256, 256)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    threading.Thread(target=os.remove, args=(picture_path,))
    image_tens = torch.reshape(transform(img), (1, 3, 256, 256))
    predictions = model(image_tens)
    predictions = torch.sigmoid_(predictions.data)
    return make_text_prediction(predictions, user_id)

def predict_state(picture_path, user_id):
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048, 3)
    model.to("cpu")
    model.load_state_dict(torch.load(r"C:\Users\Даниил\Desktop\КТ1\pythonProject1\Models\StateClassifier.pth", map_location=torch.device('cpu')))
    model.eval()
    img = Image.open(picture_path)
    transform = Compose(
        [
            Resize((512, 512)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    threading.Thread(target=os.remove, args=(picture_path,))
    image_tens = torch.reshape(transform(img), (1, 3, 512, 512))
    predictions = model(image_tens)
    predictions = predictions.data
    return make_text_state_prediction(torch.argmax(predictions).item(), user_id)

def prepare_model_detection():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.to("cpu")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(r"C:\Users\Даниил\Desktop\КТ1\pythonProject1\Models\AcneDetection.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def make_detection(picture_path, message_id, user_id):
    model = prepare_model_detection()

    pil_image = Image.open(picture_path).convert('RGB')
    open_cv_image = np.array(pil_image)
    image = open_cv_image[:, :, ::-1].copy()

    orig_image = image.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= 0.25].astype(np.int32)
        draw_boxes = boxes.copy()

        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(orig_image, 'acne', (int(box[0]), int(box[1] - 5)),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return_path = r"C:\\Users\\Даниил\\Desktop\\КТ1\\pythonProject1\\Detected_Photos\\" + message_id + '.jpg'
    im_pil.save(return_path)
    time.sleep(2)
    prediction = detection_prediction_text(draw_boxes, user_id)
    return return_path, prediction

