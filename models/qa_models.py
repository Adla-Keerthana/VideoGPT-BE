from pydantic import BaseModel
from typing import List


class QAResponse(BaseModel):
    question: str
    answer: str


class QARequest(BaseModel):
    query: str
    video_transcript: str
