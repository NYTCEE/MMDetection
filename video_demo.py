import cv2
import os
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

# Model initialization
config_file = './configs/scnet/scnet_r50_fpn_1x_coco.py'
checkpoint_file = './checkpoints/scnet_r50_fpn_1x_coco-c3f09857.pth'
device = 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

# Video input path - make sure this path is correct
video_path = 'demo/art.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Create output directory if it doesn't exist
output_dir = 'demo/output'
os.makedirs(output_dir, exist_ok=True)

# Set output path with full path
output_path = os.path.join(output_dir, 'seg_art.mp4')
print(f"Output will be saved to: {output_path}")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Object detection
    result = inference_detector(model, frame)
    
    # Visualization
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'result',
        frame,
        data_sample=result,
        draw_gt=False,
        wait_time=0
    )
    
    # Write frame to video
    output_frame = visualizer.get_image()
    video_writer.write(output_frame)
    
    frame_count += 1
    if frame_count % 10 == 0:  # Print progress every 10 frames
        print(f"Processed {frame_count} frames")

# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total frames processed: {frame_count}")
print(f"Video saved to: {output_path}")