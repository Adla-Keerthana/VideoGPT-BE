from fastapi import FastAPI
from api import video, qa, summarize

app = FastAPI()

# Include the different routes
# app.include_router(video.router, prefix="/video", tags=["Video"])
# app.router("/video")
app.include_router(video.router)
# app.include_router(qa.router, prefix="/qa", tags=["Question Answering"])
# app.include_router(summarize.router, prefix="/summarize", tags=["Summarization"])
