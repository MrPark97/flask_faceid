from flask import Flask
from flask import request
import staff
import cameras
import face_id
import threading
import cv2
import datetime
import time
import entries
from flask import render_template

app = Flask(__name__)


database = {}


def prepare_database():
    database = {}

    # load staff embeddings from db
    staff_rows = staff.get_staff_embeddings()
    for staff_row in staff_rows:
        database[staff_row[0]] = staff_row[1]

    return database



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.json)
        return face_id.face_identification(request.json)
    else:
        return render_template('entries.html', title='Home', entries=entries.get_last_entries())


def recognize_stream(camera_id=1, address='rtsp://admin:admin123@192.168.1.108:554/live'):
    vs = cv2.VideoCapture(address)

    ret, img = vs.read()

    while True:
        ret, img = vs.read()
        if ret == False:
            break

        face_id.face_recognition(img, camera_id)

    print(datetime.datetime.now())
    vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    entries.get_last_entries()

    database = prepare_database()
    all_cameras = cameras.get_all_cameras()

    threads = []
    for camera in all_cameras:
        t = threading.Thread(target=recognize_stream, args=(camera[0], camera[1]))
        threads.append(t)
        t.start()

    app.run()
