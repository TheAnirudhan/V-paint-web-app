from flask import Flask,render_template, Response
from camera import vPaint

app = Flask(__name__)

def gen(camera):
    while True:
        frame = camera.get_board()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                 + frame + b'\r\n\r\n')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/vid_feed')
def vid_feed():
    return Response(gen(vPaint()),
                    mimetype='multipart/x-mixed-replace; boundary=frame' )



if __name__ == '__main__':
    app.run(debug=True)