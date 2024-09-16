import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
# print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'face_reid') not in sys.path:
    sys.path.append(str(ROOT / 'face_reid'))
if str(ROOT / 'art2real') not in sys.path:
    sys.path.append(str(ROOT / 'art2real'))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from face_reid.main import frame_processor
from art2real.test import art2real

app = Flask(__name__)

# Directory to store uploaded sketches

UPLOAD_FOLDER = 'static/testA'
VIDEO_UPLOAD_FOLDER = 'static/videos'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)

video_file_path = None

@app.route('/', methods=['GET', 'POST'])

def index():
    global video_file_path

    if request.method == 'POST':

        # Get case number

        case_number = request.form['case_number']

        # Get uploaded sketch

        if 'sketch' in request.files:

            sketch = request.files['sketch']

            sketch_filename = f"{case_number}"
            file_pth = f'{UPLOAD_FOLDER}/{sketch_filename}.jpg'
        
        video_option = request.form['video_option']
        if video_option == 'upload':
            if 'video_file' in request.files:
                video = request.files['video_file']
                # video_filename = f"{video.filename}"
                video_file_path = os.path.join(VIDEO_UPLOAD_FOLDER, video.filename)
                # video.save(video_file_path)
 
        else:
            video_file_path = None

        if request.form['action'] == 'submit':
            sketch.save(file_pth)  # Save the sketch file
            art2real()
            os.remove(file_pth)
            if video_file_path is not None:
                video.save(video_file_path)
            # Perform Submit action: redirect to next page for processing

            return redirect(url_for('video_feed'))

        elif request.form['action'] == 'next_case':

            sketch.save(file_pth)  # Save the sketch file
            art2real()
            os.remove(file_pth)
            if video_file_path is not None:
                video.save(video_file_path)
            # Perform Next Case action: save and return to the same page

            return redirect(url_for('index'))

    return render_template('index.html')

# Route to simulate waiting for AI-generated image

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# Function to process video stream and send frames
def generate_frames():
    global video_file_path
    if video_file_path is not None:
        cap = cv2.VideoCapture(video_file_path)
    else:
        cap = cv2.VideoCapture(0)  # Capture video from webcam
    i = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Convert the frame to JPEG format for streaming
        if i%5 == 0:
            frame = frame_processor(frame, 'face_reid/faces')
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # Stream the frame to the web page
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    if video_file_path is not None:
        os.remove(video_file_path)
@app.route('/video_stream')
def video_stream():
   return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,port=8050, host='0.0.0.0')
 
