import os
import uuid
import glob
import cv2
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
import base64
import subprocess


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

clApp = ClientApp()

# Helper: get latest YOLO output image (handles exp/, exp2/, etc.)
def get_latest_output_image(original_filename):
    files = sorted(glob.glob("yolov5/runs/detect/exp*/" + os.path.basename(original_filename)), key=os.path.getmtime, reverse=True)
    return files[0] if files else None

@app.route("/")
def home():
    return render_template("index.html")

# Upload image for prediction
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../inputImage.jpg")

        output_path = get_latest_output_image(clApp.filename)
        if output_path:
            opencodedbase64 = encodeImageIntoBase64(output_path)
            result = {"image": opencodedbase64.decode('utf-8')}
        else:
            result = {"error": "Detection failed"}

        os.system("rm -rf yolov5/runs")
        return jsonify(result)

    except Exception as e:
        print(e)
        return Response("Prediction failed", status=500)

# Webcam video stream
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Capture frame from webcam and run detection
@app.route('/capture', methods=['POST'])
def capture_image():
    try:
        # Get the base64 image data from the request body
        data = request.get_json() 
        image_data = data['image']

        if not image_data:
            return jsonify({"error": "No image data found"}), 400

        # Check if the image data starts with 'data:image/jpeg;base64,'
        #if image_data.startswith("data:image/jpeg;base64,"):
            # Remove the prefix and decode the base64 string
            #image_data = image_data.split(",")[1]
        #else:
            #return jsonify({"error": "Invalid image data format"}), 400
        
        print(f"Received image data: {image_data[:30]}...")  # Print first few characters

        # Decode base64 image data
        
        binary_data = base64.b64decode(image_data)

        # Generate a unique filename using uuid to avoid collisions


        unique_filename = f"webcam_{uuid.uuid4()}.jpg"
        relative_path = os.path.join("data", unique_filename)  # relative path for YOLOv5
        absolute_path = os.path.join("yolov5", relative_path)  # full path from Flask

        # ✅ Make sure the directory exists
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        # ✅ Save the image to yolov5/data/
        with open(absolute_path, "wb") as f:
            f.write(binary_data)

        # Run the YOLOv5 detection using subprocess
        result = subprocess.run(
            ['python', 'detect.py', '--weights', 'best.pt', '--img', '640', '--conf', '0.25', '--source', relative_path],
            cwd='yolov5',
            capture_output=True,
            text=True
        )


        # Check if the process ran successfully
        if result.returncode != 0:
            print(result.stderr)
            return jsonify({"error": "Prediction failed"}), 500

        # Load the output image after YOLOv5 detection
        output_path = get_latest_output_image(absolute_path)

        if not output_path or not os.path.exists(output_path):
            return jsonify({"error": "No output image generated"}), 500


        # Open the result image and convert to base64
        with open(output_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Convert the image to base64 to send it back to the frontend
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')

        # Optionally clean up the captured image after prediction
        os.remove(absolute_path)  # Uncomment this line to delete the file after prediction
        
        
        # Return the result of the prediction, sending the base64-encoded image back
        return jsonify({"message": "Prediction complete", "image": encoded_img})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080,debug=True)
