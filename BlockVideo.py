import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def split_video(input_video_path, output_folder, rows=5, cols=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}, Total Frames: {total_frames}")

    block_width = frame_width // cols
    block_height = frame_height // rows

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    def process_block(r, c):
        block_output_path = os.path.join(output_folder, f"block_{r}_{c}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(block_output_path, fourcc, fps, (block_width, block_height))

        for frame in frames:
            block_frame = frame[r * block_height:(r + 1) * block_height,
                          c * block_width:(c + 1) * block_width]
            out.write(block_frame)
        out.release()

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for r in range(rows):
            for c in range(cols):
                futures.append(executor.submit(process_block, r, c))

        for future in futures:
            future.result()

    print(f"Video split into {rows * cols} blocks and saved to {output_folder}.")


