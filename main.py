import insightface
import time
import datetime
import os
import cv2
import numpy as np
import glob
import urllib, base64
import threading
import cameras
import logs
import staff
import urllib.parse

from http.server import HTTPServer, BaseHTTPRequestHandler

from io import BytesIO

import multipart
from wsgiref.simple_server import make_server

SLEEP_TIME = 3

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# choose face detection model: retinaface_r50_v1 or retinaface_mnet025_v2
model = insightface.app.FaceAnalysis(det_name='retinaface_mnet025_v2', rec_name='arcface_r100_v1', ga_name=None)
model.prepare(ctx_id=0, nms=0.4)
database = {}
staff_ignoring = {}
last_frame = {}


def simple_app(environ, start_response):
    fields = {}
    files = {}

    def on_field(field):
        fields[field.field_name] = field.value

    def on_file(file):
        files[file.field_name] = {'name': file.file_name, 'file_object': file.file_object}

    multipart_headers = {'Content-Type': environ['CONTENT_TYPE']}
    multipart_headers['Content-Length'] = environ['CONTENT_LENGTH']
    multipart.parse_form(multipart_headers, environ['wsgi.input'], on_field, on_file)
    print(urllib.parse.unquote(fields[b'image'].decode('utf-8')))
    img = cv2.imread(urllib.parse.unquote(fields[b'image'].decode('utf-8')))
    cv2.imwrite('tes.jpg', img)
    face = model.get(img)[0]
    face_embedding = face.normed_embedding
    face_box = face.bbox.astype(np.int).flatten()
    s_emb = str(list(face_embedding))
    s_box = str(list(face_box))
    content = '{"box": ' + s_box + ', "embedding": ' + s_emb + '}'

    content = [content.encode('utf-8')]
    status = '200 OK'
    headers = [('Content-type', 'application/json; charset=utf-8')]
    start_response(status, headers)
    return content


httpd = make_server('0.0.0.0', 8000, simple_app)


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


def ignore_staff(camera_id, staff_id):
    staff_ignoring[camera_id][staff_id] = 1
    time.sleep(SLEEP_TIME)
    staff_ignoring[camera_id][staff_id] = 0


def prepare_database_files():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("database/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = model.get(cv2.imread(file))[
            0].normed_embedding  # embedding_norm, embedding, normed_embedding

        f = open("demofile2.txt", "a")
        f.write("%s\n" % (identity))
        f.write("{")
        for embed in database[identity]:
            f.write("%f, " % (embed))
        f.write("}")
        f.write("\n\n")
        f.close()

    return database


def prepare_database():
    database = {}

    # load staff embeddings from db
    staff_rows = staff.get_staff_embeddings()
    for staff_row in staff_rows:
        database[staff_row[0]] = staff_row[1]

    return database


def process_frame(img, camera_id=1):
    faces = model.get(img, det_thresh=0.8, det_scale=1.0, max_num=0)
    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(np.int).flatten()

        identity = who_is_it(face.normed_embedding, camera_id)
        if identity is None:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        else:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
            img = putText(img, identity, bbox[2], bbox[3])
    return img


def process_last_frame(camera_id=1):
    while True:
        cv2.imshow('Video', process_frame(last_frame[camera_id], camera_id))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def who_is_it(normed_embedding, camera_id=1):
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
        if staff_ignoring[camera_id][identity] == 0:
            logs.insert_entry(camera_id, float(max_sim), identity)
            x = threading.Thread(target=ignore_staff, args=(camera_id, identity))
            x.start()
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

    all_cameras = cameras.get_all_cameras()

    for camera in all_cameras:
        staff_ignoring[camera[0]] = {}
        for staff_id in database:
            staff_ignoring[camera[0]][staff_id] = 0
        print("id:", camera[0], "url:", camera[1])

    # recognize_image('Sanjar2.png')
    # recognize_video()

    x = threading.Thread(target=httpd.serve_forever)
    x.start()

    # recognize_stream('rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0')
    recognize_stream()

    # logs.insert_entry(1, 0.5, 2)

    # sotrudniki = staff.get_staff_embeddings()


