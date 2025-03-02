import torch
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ffmpeg
import numpy as np
import faiss
import open_clip
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline


# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt")
clip_model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(device)
action_recognition = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=0 if torch.cuda.is_available() else -1,
)

#Extract frames from video
def extract_frames(video_path, output_folder, fps=1):
    print("Extracting frames...")
    print("output_folder", output_folder)
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    ffmpeg_path = r"C:\Users\devad\Downloads\ffmpeg-7.1-essentials_build\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"
    os.makedirs(output_folder, exist_ok=True)
    (
        ffmpeg.input(video_path)
        .output(f"{output_folder}/frame_%04d.jpg", vf=f"fps={fps}")
        .run(cmd=ffmpeg_path, overwrite_output=True)
    )


def detect_objects(frame_path):
    print("Detecting objects...")
    frame = cv2.imread(frame_path)
    results = yolo_model(frame)
    objects_detected = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            label = result.names[int(cls)]
            objects_detected.append(label)
    return objects_detected


def recognize_action(frame_path):
    print("Recognizing action...")
    image = Image.open(frame_path).convert("RGB")
    predictions = action_recognition(image)
    return predictions[0]["label"]


def extract_scene_features(frame_path):
    print("Extracting scene features...")
    image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image).cpu()
    return features.cpu().numpy().flatten()


def process_video(video_path, output_folder="frames", fps=1):
    print("Processing video...")
    extract_frames(video_path, output_folder, fps)
    metadata = []
    vectors = []
    for frame in tqdm(sorted(os.listdir(output_folder))):
        frame_path = os.path.join(output_folder, frame)
        objects = detect_objects(frame_path)
        action = recognize_action(frame_path)
        scene_vector = extract_scene_features(frame_path)
        metadata.append(
            {
                "frame": frame_path,
                "objects": objects,
                "action": action,
                "vector": scene_vector,
            }
        )
        vectors.append(scene_vector)
    return metadata, np.array(vectors)


def build_faiss_index(vectors):
    print("Building index...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def match_query(query, metadata, index):
    print("Matching query...")
    text_vector = (
        clip_model.encode_text(tokenizer([query]).to(device)).detach().cpu().numpy()
    )
    distances, indices = index.search(text_vector, k=5)
    return [metadata[i]["frame"] for i in indices[0]]


def extract_clips(
    matched_frames, video_path, output_video="output_clip.mp4", clip_duration=5
):
    print("Extracting clips...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_clips = []
    for frame_path in matched_frames:
        timestamp = int(frame_path.split("_")[-1].split(".")[0])
        start_frame = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        temp_output = f"clip_{timestamp}.mp4"
        out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
        for _ in range(int(clip_duration * fps)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        output_clips.append(temp_output)
    cap.release()
    merge_clips(output_clips, output_video)
    return output_video


def merge_clips(input_clips, output_video):
    print("Merging clips...")
    if not input_clips:
        raise ValueError("No clips to merge.")
    first_clip = cv2.VideoCapture(input_clips[0])
    frame_width = int(first_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = first_clip.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    first_clip.release()
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    for clip in input_clips:
        cap = cv2.VideoCapture(clip)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    out.release()
