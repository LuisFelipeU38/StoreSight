import os
import uuid
import cv2

def extract_frames(video_path, images_folder):
    os.makedirs(images_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        image_filename = f"{uuid.uuid4()}.jpg"
        cv2.imwrite(os.path.join(images_folder, image_filename), frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames and saved in {images_folder}.")

def main():
    video_path = input("Enter the path to the video: ")
    images_folder = os.path.join("uploads", "extracted_images")
    
    extract_frames(video_path, images_folder)
    
    print("Creating labels for images in uploads/extracted_images.")
    print("Please label the images using MakeSense.ai.")
    print(f"Once completed, save your annotations in the folder: {os.path.join(images_folder, 'annotations')}.")
    
    # You can now proceed to train your model with the images
    # Uncomment the following line if you have a function to train your model
    # train_model(images_folder)

if __name__ == "__main__":
    main()
