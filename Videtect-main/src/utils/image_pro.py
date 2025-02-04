# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
import tensorflow
import cv2,sys
import numpy as np

from numpy import expand_dims
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import os

class ImagePro:
    # Define a function to extract frames from the video:
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
            image_path = os.path.join(output_folder, f"{frame_counter}.jpg")
            cv2.imwrite(image_path, frame)

            image_counter += 1

            # Display progress
            progress = (frame_counter / total_frames) * 100
            print(f"Progress: {progress:.2f}%")

        # Release the video capture object
        video.release()

    # Define a function that takes a list of images as input:
    def process_images():
        #for image in images:
            # Process each image
            # Example: print the image path
            print("image")


    def extract_text_from_image(image):
    # Use pytesseract to extract text from the image
    img = Image.fromarray(image)
    text = pytesseract.image_to_string(img, config='--psm 11')  # PSM 11 treats the image as a single line of text
    return text.strip()


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores, folder_path, j, line_y):
    # load the image
    image = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(image)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # Ensure coordinates are within the valid range
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(image.shape[0], y2), min(image.shape[1], x2)
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = f"{v_labels[i]} ({v_scores[i]:.3f})"
        pyplot.text(x1, y1, label, color='white')

        # Draw a line at a specific y-coordinate (e.g., line_y)
        image = cv2.line(image, (0, line_y), (image.shape[1], line_y), (255, 0, 0), 2)
        pyplot.imshow(image)

        # Draw rectangles based on the box coordinates
        for box in v_boxes:
            boxy = (box.ymin + box.ymax) / 2
            if boxy > line_y:
                rect = pyplot.Rectangle((box.xmin, box.ymin), box.xmax - box.xmin, box.ymax - box.ymin,
                                        fill=False, color='r')
            else:
                rect = pyplot.Rectangle((box.xmin, box.ymin), box.xmax - box.xmin, box.ymax - box.ymin,
                                        fill=False, color='g')

            # Perform number plate recognition on the region of interest (ROI)
            roi_image = image[max(0, int(box.ymin)):min(image.shape[0], int(box.ymax)),
                             max(0, int(box.xmin)):min(image.shape[1], int(box.xmax))]
            number_plate_text = extract_text_from_image(roi_image)

            print(f"Number Plate Text: {number_plate_text}")

            pyplot.gca().add_patch(rect)

    # Save the figure
    pyplot.savefig(os.path.join(folder_path, f'image_{j}.png'))
    # Display the plot
    pyplot.show()
 	#pyplot.savefig(folder_path)


"""
  # Show the plot
  # pyplot.show()
	pyplot.show()
 	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	ax.plot(range(100))
	fig.savefig(filename)"""