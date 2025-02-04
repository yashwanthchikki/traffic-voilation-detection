def extract_frames(video_path, output_folder, frame_rate):
        video = cv2.VideoCapture(video_path)

        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the interval to extract frames based on the desired frame rate
        frame_interval = int(video.get(cv2.CAP_PROP_FPS) / frame_rate)

        # Initialize variables
        frame_counter = 0
        image_counter = 0

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Extract frames from the video
        while video.isOpened():
            # Read the next frame
            ret, frame = video.read()

            if not ret:
                break

            frame_counter += 1

            # Skip frames if necessary
            if frame_counter % frame_interval != 0:
                continue

            # Save the frame as an image
            image_path = os.path.join(output_folder, f"frame_{image_counter}.jpg")
            cv2.imwrite(image_path, frame)


            image_counter += 1

            # Display progress
            progress = (frame_counter / total_frames) * 100
            print(f"Progress: {progress:.2f}%")

        # Release the video capture object
        video.release()
def process_images(image):
        for image in images:
            # Process each image
            # Example: print the image path
            print(image)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images



video_path = "red.mp4"
output_folder = "output"
output="final"
frame_rate = 2

extract_frames(video_path,output_folder,frame_rate)
#process_images(image)

# load yolov3 model
directory = output_folder
j=0
folder_path = directory
file_prefix = "frame_"
file_extension = ".jpg"
files = os.listdir(folder_path)
filtered_files = [file_name for file_name in files if file_name.startswith(file_prefix) and file_name.endswith(file_extension)]
sorted_files = sorted(filtered_files, key=lambda x: int(x[len(file_prefix):-len(file_extension)]))

for file_name in sorted_files:
	full_path = os.path.join(folder_path, file_name)
	photo_filename = os.path.join(directory,file_name)
	# define the expected input shape for the model
	input_w, input_h = 416, 416
	# define our new photo
	# photo_filename ='/content/output/frame_0.jpg'
	# load and prepare image
	image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
	# make prediction
	yhat = model.predict(image)
	# summarize the shape of the list of arrays
	print([a.shape for a in yhat])
	# define the anchors
	anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
	# define the probability threshold for detected objects
	class_threshold = 0.6
	boxes = list()
	for i in range(len(yhat)):
		# decode the output of the network
		boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
	# correct the sizes of the bounding boxes for the shape of the image
	correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
	# suppress non-maximal boxes
	do_nms(boxes, 0.5)
	# define the labels
	labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
		"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
		"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
		"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
		"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
		"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
		"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
		"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
		"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
	# get the details of the detected objects
	v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
	# summarize what we found
	for i in range(len(v_boxes)):
		print(v_labels[i], v_scores[i])
	# draw what we found
	j+=1
	draw_boxes(photo_filename,v_boxes, v_labels, v_scores,output,j,line_y=460)
      


import os
import cv2
def generate_video():
    image_folder = "C://Users//kisho//OneDrive//Desktop//def//final"
    video_name = "C://Users//kisho//OneDrive//Desktop//def//myvideo.avi"
    os.chdir("C://Users//kisho//OneDrive//Desktop//def//final")

    images = [img for img in sorted(os.listdir(image_folder))
                    if img.endswith(".jpg") or
                      img.endswith(".jpeg") or
                      img.endswith("png")]#I'll use my own function for that, just easier to read

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 0.25, (width, height))
    print(video)
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    #video.release()

generate_video()