from fastapi import APIRouter, File, UploadFile
from services.video_service import (
    process_video,
    build_faiss_index,
    match_query,
    extract_clips,
)
from fastapi.responses import FileResponse
import os

router = APIRouter()


@router.post("/extract_clips")
async def upload_video(
    file: UploadFile = File(...), query: str = "Find a specific action in video"
):
    os.makedirs("uploads", exist_ok=True)  # Ensure upload folder exists
    file_path = os.path.join("uploads", file.filename)
    # file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    metadata, vectors = process_video(file_path, fps=1)
    index = build_faiss_index(vectors)
    matched_frames = match_query(query, metadata, index)
    output_video = extract_clips(matched_frames, file_path)
    return FileResponse(
        output_video, media_type="video/mp4", filename="extracted_clip.mp4"
    )
