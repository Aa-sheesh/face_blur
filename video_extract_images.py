import cv2
import os

def images_to_video(image_folder, output_video_path, fps=23.8):
    # Get list of image files sorted by name
    images = sorted([
        img for img in os.listdir(image_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not images:
        raise ValueError("No images found in the specified folder.")

    # Read the first image to get frame dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame.shape[0:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage
images_to_video(r"C:\Users\Aashish\OneDrive\Desktop\face_blur\Trimmed Videos\VFS Face Blur trimmed Video 3_frames_blurred", "output_video.mp4", fps=23.8)
# wait for some time
# ok sir 