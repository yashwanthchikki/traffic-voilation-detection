import os
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pytesseract
from datetime import datetime
import pickle
from tqdm import tqdm  # Import tqdm for progress bar

# Load the YOLOv3 model
with open('trained_yolov3_model.pkl', 'rb') as f:
    model = pickle.load(f)

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size=shape)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, 0)
    return image, width, height

def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    return v_boxes, v_labels, v_scores

def extract_text_from_image(image):
    img = Image.fromarray(image)
    text = pytesseract.image_to_string(img, config='--psm 11')
    return text.strip()

def draw_boxes(filename, v_boxes, v_labels, v_scores, folder_path, j, line_y):
    image = plt.imread(filename)
    plt.imshow(image)
    ax = plt.gca()
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(image.shape[0], y2), min(image.shape[1], x2)
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        ax.add_patch(rect)
        label = f"{v_labels[i]} ({v_scores[i]:.3f})"
        plt.text(x1, y1, label, color='white')
        image = cv2.line(image, (0, line_y), (image.shape[1], line_y), (255, 0, 0), 2)
        plt.imshow(image)
        for box in v_boxes:
            boxy = (box.ymin + box.ymax) / 2
            if boxy > line_y:
                rect = plt.Rectangle((box.xmin, box.ymin), box.xmax - box.xmin, box.ymax - box.ymin,
                                     fill=False, color='r')
            else:
                rect = plt.Rectangle((box.xmin, box.ymin), box.xmax - box.xmin, box.ymax - box.ymin,
                                     fill=False, color='g')

            roi_image = image[max(0, int(box.ymin)):min(image.shape[0], int(box.ymax)),
                             max(0, int(box.xmin)):min(image.shape[1], int(box.xmax))]
            number_plate_text = extract_text_from_image(roi_image)
            print(f"Number Plate Text: {number_plate_text}")
            plt.gca().add_patch(rect)
    plt.savefig(os.path.join(folder_path, f'image_{j}.png'))
    plt.show()

def extract_frames(video_path, output_folder, frame_rate):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize frame count
    frame_count = 0

    # Read and save frames at the specified frame rate
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count // frame_rate}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object and close the video file
    cap.release()
    cv2.destroyAllWindows()

def generate_video(input_folder, output_video_path, frame_rate):
    # Get the list of frame filenames
    frame_filenames = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]

    # Sort the frame filenames
    frame_filenames.sort()

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame = cv2.imread(frame_filenames[0])
    height, width, _ = frame.shape
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write frames to the video
    for filename in frame_filenames:
        frame = cv2.imread(filename)
        out.write(frame)

    # Release the video writer
    out.release()

def count_vehicles(v_labels):
    # Count the number of vehicles detected in each category
    vehicle_counts = {}
    for label in v_labels:
        vehicle_counts[label] = vehicle_counts.get(label, 0) + 1
    return vehicle_counts

# Define constants and variables
anchors = ...
obj_thresh = ...
net_h, net_w = ...
nms_thresh = ...
thresh = ...
labels = ...

# Define the directory and other variables
video_path = "video.mp4"
output_folder = "output_frames"
output_folder_final = "final_output"
frame_rate = 2

# Extract frames from the video
extract_frames(video_path, output_folder, frame_rate)

# Process each frame for object detection
directory = output_folder
j = 0
folder_path = directory
file_prefix = "frame_"
file_extension = ".jpg"
files = os.listdir(folder_path)
filtered_files = [file_name for file_name in files if file_name.startswith(file_prefix) and file_name.endswith(file_extension)]
sorted_files = sorted(filtered_files, key=lambda x: int(x[len(file_prefix):-len(file_extension)]))

# Store detected vehicle information
detected_vehicles = []

# Define progress bar
with tqdm(total=len(sorted_files), desc='Processing Frames') as pbar:
    for file_name in sorted_files:
        # Load and preprocess image
        full_path = os.path.join(folder_path, file_name)
        photo_filename = os.path.join(directory, file_name)
        input_w, input_h = 416, 416
        image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

        # Make prediction
        yhat = model.predict(image)

        # Decode and process YOLOv3 output
        boxes = decode_netout(yhat[0], anchors, obj_thresh, net_h, net_w)
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
        do_nms(boxes, nms_thresh)

        # Draw bounding boxes and extract number plate text
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, thresh)
        draw_boxes(photo_filename, v_boxes, v_labels, v_scores, folder_path, j, line_y)

        for label, box in zip(v_labels, v_boxes):
            detected_vehicles.append((label, box))

        j += 1

        pbar.update(1)  # Update progress bar

# Generate final video
output_video_path = "output_video.avi"
generate_video(output_folder, output_video_path, frame_rate)

# Count vehicles detected
vehicle_counts = count_vehicles([v_label for _, v_label, _ in detected_vehicles])

# Visualize vehicle count graph
labels = vehicle_counts.keys()
counts = vehicle_counts.values()

plt.bar(labels, counts)
plt.xlabel('Vehicle Category')
plt.ylabel('Count')
plt.title('Vehicle Count by Category')
plt.xticks(rotation=45)
plt.show()
"""
# Document the date and time with the number plate and category information
now = datetime.now()
date_time = now.strftime("%Y-%m-%d %H:%M:%S")

with open('detected_vehicles.txt', 'w', encoding='utf-8') as f:
    f.write("Date & Time\tNumber Plate\tCategory\n")
    for plate, label in detected_vehicles:
        f.write(f"{date_time}\t{plate}\t{label}\n")"""
