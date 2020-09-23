import sys

sys.path.append('MTCNN')
from torchvision import transforms as trans
from face_model import MobileFaceNet
from utils.align_trans import *
from MTCNN import create_mtcnn_net
from flask import jsonify

import psycopg2
import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
detect_model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
print('MobileFaceNet face detection model generated')
detect_model.eval()

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def face_recognition(img, camera):
    try:
        bboxes, landmarks = create_mtcnn_net(img, 20, device,
                                             p_model_path='MTCNN/weights/pnet_Weights',
                                             r_model_path='MTCNN/weights/rnet_Weights',
                                             o_model_path='MTCNN/weights/onet_Weights')

        faces = Face_alignment(img, default_square=True, landmarks=landmarks)
    except:
        return

        embeddings = []
        conn = None

    try:
        params = config.db_config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        i = 0

        with torch.no_grad():
            for face in faces:
                i += 1
                cv2.imwrite(str(i)+".jpg", face)
                embedding = detect_model(test_transform(face).to(device).unsqueeze(0))

                cur.execute(
                    'INSERT INTO entries (entry_embedding, entry_camera) VALUES (%s, %s)',
                    (embedding.squeeze().tolist(), camera))

        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def face_identification(request):
    image_path = request['path']

    img = cv2.imread(image_path)

    bboxes, landmarks = create_mtcnn_net(img, 20, device,
                                         p_model_path='MTCNN/weights/pnet_Weights',
                                         r_model_path='MTCNN/weights/rnet_Weights',
                                         o_model_path='MTCNN/weights/onet_Weights')

    faces = Face_alignment(img, default_square=True, landmarks=landmarks)

    embeddings = []

    with torch.no_grad():
        for face in faces:
            embedding = detect_model(test_transform(face).to(device).unsqueeze(0))
            embeddings.append(embedding.squeeze().tolist())

    return jsonify(
        embeddings=embeddings
    )
