import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np
import torch
import torchvision.transforms as T

# Buttons:
#     1 - use function to display image as background with segmentator:
#         a - previous image
#         d - next image
#         l - leave a function
#     2 - use function to display image as background with AI pretrained model:
#         l - leave a function
#     3 - use function to blur background AI pretrained model:
#         l - leave a function
#     4 - use function to blur background with Background Substractor:
#         l - leave a function
#     q - end program

def replace_background_segmentator(image_index):
    cv2.destroyAllWindows()
    while True:
        _, frame = capture.read()
        frame_without_background = segmentator.removeBG(frame, image_list_background[image_index], threshold=0.1)
        window_merge = cvzone.stackImages([frame, frame_without_background], 2, 1)
        fps, window_merge = fps_counter.update(window_merge, color=(0, 255, 0))

        cv2.imshow("Background as image with segmentator", window_merge)
        image_key = cv2.waitKey(1)
        if image_key == ord('a'):
            if image_index > 0:
                image_index -= 1
        elif image_key == ord('d'):
            if image_index < len(image_list_background) - 1:
                image_index += 1
        elif image_key == ord('l'):
            break
    cv2.destroyAllWindows()

def skin_color_mask(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for skin color in the HSV color space
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Apply morphological operations to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return skin_mask

def blur_background_substractor():
    cv2.destroyAllWindows()
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    while True:
        _, frame = capture.read()

        # Apply the background subtractor
        fg_mask = background_subtractor.apply(frame)

        # Create a skin color mask
        skin_mask = skin_color_mask(frame)

        # Combine the foreground mask and the skin color mask
        combined_mask = cv2.bitwise_or(fg_mask, skin_mask)

        # Blur the background
        blurred_background = cv2.GaussianBlur(frame, (49, 49), 0)

        # Merge the person and the blurred background
        result = frame.copy()
        result[combined_mask == 0] = blurred_background[combined_mask == 0]

        cv2.imshow('Blurred Background by substractor', result)

        blur_key = cv2.waitKey(1)
        if blur_key == ord('l'):
            break

    cv2.destroyAllWindows()

def blur_background_AI():
    cv2.destroyAllWindows()

    # Define the transform
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((260, 260)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_count = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error reading frame from webcam")
            return

        frame_count += 1

        if frame_count % 4 != 0:
            continue

        # Apply the transform to the frame
        input_tensor = transform(frame).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Create a mask of the person
        person_mask = (output_predictions == 15)  # 15 is the class index for 'person' in the COCO dataset

        # Resize the mask to the original frame size
        person_mask = cv2.resize(person_mask.astype('float32'), (frame.shape[1], frame.shape[0]))

        # Threshold the mask
        _, person_mask = cv2.threshold(person_mask, 0.5, 1, cv2.THRESH_BINARY)

        # Create a 3-channel mask
        mask_3 = np.repeat(person_mask[:, :, None], 3, axis=2)

        blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)

        # Combine the blurred background and the unblurred person
        combined_frame = blurred_frame * (1 - mask_3) + frame * mask_3

        cv2.imshow('Background as image with AI', combined_frame.astype('uint8'))

        if cv2.waitKey(1) & 0xFF == ord('l'):
            break

    cv2.destroyAllWindows()

def image_background_AI():
    cv2.destroyAllWindows()

    # Define the transform
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((260, 260)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    background_image = cv2.imread('mchtr.png')
    frame_count = 0
    while True:
        # Read the frame from the webcam
        ret, frame = capture.read()
        if not ret:
            print("Error reading frame from webcam")
            return

        frame_count += 1
        if frame_count % 4 != 0:
            continue

        # Resize the background image to match the frame
        resized_background = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

        # Apply the transform to the frame and add an extra batch dimension
        input_tensor = transform(frame).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Create a mask of the person
        person_mask = (output_predictions == 15)  # 15 is the class index for 'person' in the COCO dataset

        # Resize the mask to the original frame size
        person_mask = cv2.resize(person_mask.astype('float32'), (frame.shape[1], frame.shape[0]))

        # Threshold the mask
        _, person_mask = cv2.threshold(person_mask, 0.5, 1, cv2.THRESH_BINARY)

        # Create a 3-channel mask
        mask_3 = np.repeat(person_mask[:, :, None], 3, axis=2)

        # Combine the background image and the person
        combined_frame = resized_background * (1 - mask_3) + frame * mask_3

        cv2.imshow('Blured Background AI', combined_frame.astype('uint8'))

        if cv2.waitKey(1) & 0xFF == ord('l'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.eval()

    segmentator = SelfiSegmentation();
    fps_counter = cvzone.FPS()

    list_images = os.listdir("images")
    image_list_background = []
    image_index = 0

    for image_Path in list_images:
        image = cv2.imread(f'images/{image_Path}')
        image_list_background.append(image)

    while True:
        _, frame = capture.read()
        cv2.imshow("Default Webcam", frame)

        key = cv2.waitKey(1)
        if key == ord('1'):
            replace_background_segmentator(image_index)
        elif key == ord('2'):
            image_background_AI()
        elif key == ord('3'):
            blur_background_AI()
        elif key == ord('4'):
            blur_background_substractor()
        elif key == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()











