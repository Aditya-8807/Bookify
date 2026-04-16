from typing import TypedDict, List, Dict, Optional, Any


class TranscriptSegment(TypedDict):
    start: float
    end: float
    text: str


class VideoMeta(TypedDict):
    video_id: str
    title: str
    description: str
    playlist_index: int
    ref_urls: List[str]
    audio_path: str


class Transcript(TypedDict):
    video_id: str
    title: str
    segments: List[TranscriptSegment]
    full_text: str


class CorrectedTranscript(TypedDict):
    video_id: str
    title: str
    segments: List[TranscriptSegment]
    full_text: str
    corrections: List[Dict]


class TopicGroup(TypedDict):
    name: str
    slug: str
    video_ids: List[str]
    dependency_order: int
    prerequisites: List[str]
    ref_urls: List[str]
    ref_contents: Dict[str, str]


class Citation(TypedDict):
    type: str
    source: str
    timestamp: Optional[str]
    passage: str


class VerifiedTopic(TypedDict):
    name: str
    slug: str
    prose: str
    citations: List[Citation]
    stats: Dict[str, Any]
