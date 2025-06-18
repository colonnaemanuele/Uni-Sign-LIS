import os
import cv2

def crop_video_rect(input_path, output_path, crop_x, crop_y, crop_width, crop_height):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

        if cropped.shape[0] != crop_height or cropped.shape[1] != crop_width:
            print(f"Frame fuori dai limiti: {input_path}. Skipping frame.")
            continue

        out.write(cropped)

    cap.release()
    out.release()
    print(f"Salvato: {output_path}")

def process_all_videos_in_folders(input_root, output_root, crop_x, crop_y, crop_width, crop_height):
    for subdir, dirs, files in os.walk(input_root):
        # Compute relative path to mirror folder structure
        rel_path = os.path.relpath(subdir, input_root)
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        for filename in files:
            if filename.lower().endswith('.mp4'):
                input_path = os.path.join(subdir, filename)
                output_path = os.path.join(output_dir, filename)
                print(f"\nCropping: {input_path}")
                crop_video_rect(input_path, output_path, crop_x, crop_y, crop_width, crop_height)


if __name__ == "__main__":
    input_root = "./LIS/refined/video_tagliati"
    output_root = "./LIS/refined/cropped_videos"
    crop_x = 0
    crop_y = 150
    crop_width = 720
    crop_height = 710
    print("Inizio il processo di ritaglio dei video...")
    process_all_videos_in_folders(input_root, output_root, crop_x, crop_y, crop_width, crop_height)
    print("Processamento completato.")