import cv2
import argparse
import tflite_runtime.interpreter as tflite
from BlazeFaceDetection.blazeFaceDetector import blazeFaceDetector
import time

def start_tflite(model, ext_delegate):
	scoreThreshold = 0.7
	iouThreshold = 0.3
	modelType = "front"

	# Initialize face detector
	faceDetector = blazeFaceDetector(modelType, scoreThreshold, iouThreshold, ext_delegate)

	# Initialize webcam
	camera = cv2.VideoCapture(3)
	cv2.namedWindow("detections", cv2.WINDOW_NORMAL) 
	while True:

		# Read frame from the webcam
		ret, img = camera.read()
		start = time.time()	

		# Detect faces
		detectionResults = faceDetector.detectFaces(img)

		# Draw detections
		img_plot = faceDetector.drawDetections(img, detectionResults)
                
		
		print(1.0 / (time.time() - start))
        
		cv2.imshow("detections", img_plot)

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	camera.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default=None, help='the model to use')
    parser.add_argument('--ext_delegate', default='/usr/lib/libvx_delegate.so', help='external_delegate_library_path')
    parser.add_argument('--ext_delegate_options', help='external delegate options')
    args = parser.parse_args()

    ext_delegate = None
    ext_delegate_options = {}

    # parse extenal delegate options
    if args.ext_delegate_options is not None:
        options = args.ext_delegate_options.split(';')
        for o in options:
            kv = o.split(':')
            if (len(kv) == 2):
                ext_delegate_options[kv[0].strip()] = kv[1].strip()
            else:
                raise RuntimeError('Error parsing delegate option: ' + o)

    # load external delegate
    if args.ext_delegate is not None:
        print('Loading external delegate from {} with args: {}'.format(args.ext_delegate, ext_delegate_options))
        ext_delegate = [tflite.load_delegate(args.ext_delegate, ext_delegate_options)]

        print("Went here.")
    start_tflite(args.model, ext_delegate)