import sys

sys.path.append('MTCNN')
from torchvision import transforms as trans
from face_model import MobileFaceNet
from utils.align_trans import *
from MTCNN import create_mtcnn_net
from flask import jsonify

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
detect_model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
print('MobileFaceNet face detection model generated')
detect_model.eval()

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


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
