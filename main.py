import time
import datetime
import os
import cv2
import numpy as np
import glob
import urllib
import base64
import threading
import staff
import urllib.parse
SLEEP_TIME = 3


import sys

sys.path.append('MTCNN')
from torchvision import transforms as trans
from face_model import MobileFaceNet
from utils.align_trans import *
from MTCNN import create_mtcnn_net
from flask import jsonify

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
detect_model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
print('MobileFaceNet face detection model generated')
detect_model.eval()

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

database = {}
last_frame = {}


def putText(img, text, text_offset_x, text_offset_y, font_scale=1.5):
    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)

    font = cv2.FONT_HERSHEY_PLAIN
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    img = cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    img = cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0),
                      thickness=1)

    return img

def prepare_database():
    database = {}

    # load staff embeddings from db
    staff_rows = staff.get_staff()
    for staff_row in staff_rows:
        database[staff_row[0]] = {"embedding": staff_row[3], "name": staff_row[1]}

    return database


def process_frame(img):
    try:
        bboxes, landmarks = create_mtcnn_net(img, 20, device,
                                             p_model_path='MTCNN/weights/pnet_Weights',
                                             r_model_path='MTCNN/weights/rnet_Weights',
                                             o_model_path='MTCNN/weights/onet_Weights')

        faces = Face_alignment(img, default_square=True, landmarks=landmarks)
    except:
        return img

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :4].astype(np.int).flatten()
        embedding = detect_model(test_transform(faces[i]).to(device).unsqueeze(0))

        identity = who_is_it(embedding)
        if identity is None:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        else:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
            img = putText(img, identity, bbox[2], bbox[3])
    return img


def process_last_frame(camera_id=1):
    while True:
        cv2.imshow('Video', process_frame(last_frame[camera_id]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def who_is_it(normed_embedding):
    max_sim = 0.0
    identity = None
    for (name, normed_db_embedding) in database.items():
        sim = np.dot(normed_db_embedding, normed_embedding)
        if sim > max_sim:
            max_sim = sim
            identity = name

    if max_sim <= 0.4:
        return None
    else:
        return str(identity)


def recognize_image(input_path='content/test.jpg', output_path='test.jpg'):
    cv2.imwrite(output_path, process_frame(cv2.imread(input_path)))


def recognize_video(input_path='test2.mp4', output_path="output.mp4", height=848, width=480, camera_id=0):
    vs = cv2.VideoCapture(input_path)

    ret, img = vs.read()

    global last_frame
    last_frame[camera_id] = img

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (height, width))

    while True:
        ret, img = vs.read()
        if ret == False:
            break

        last_frame[camera_id] = img
        out.write(process_frame(last_frame[camera_id]))

    print(datetime.datetime.now())
    vs.release()
    out.release()


def recognize_stream(address='rtsp://admin:123@192.168.1.108:554/live', camera_id=1):
    vs = cv2.VideoCapture(address)

    ret, img = vs.read()
    global last_frame
    last_frame[camera_id] = img

    x = threading.Thread(target=process_last_frame, args=(camera_id,))
    x.start()

    while True:
        ret, img = vs.read()
        if ret == False:
            break

        last_frame[camera_id] = img

    print(datetime.datetime.now())
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    database = prepare_database()
    recognize_stream()


