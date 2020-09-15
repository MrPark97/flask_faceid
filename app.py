from flask import Flask
from flask import request
import face_id

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.json)
        return face_id.face_identification(request.json)
    else:
        return 'Hello World!'


if __name__ == '__main__':
    app.run()
