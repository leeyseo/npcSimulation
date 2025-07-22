# data_structures.py
import datetime
import uuid
from typing import Optional, List


class Memory:
    """메모리 데이터 구조"""

    def __init__(self, memory_type: str, description: str, importance: int,
                 embedding: list[float], keywords: set, evidence_ids: list[str] = None,emotion=None, strategy=None, personality=None):
        self.id = str(uuid.uuid4())
        self.type = memory_type
        self.timestamp = datetime.datetime.now()
        self.description = description
        self.importance = importance
        self.embedding = embedding
        self.keywords = keywords
        self.last_accessed = datetime.datetime.now()
        self.evidence_ids = evidence_ids if evidence_ids else []
        self.emotion = emotion
        self.strategy = strategy
        self.personality = personality

    def __repr__(self):
        return f"Memory(ID: {self.id[-6:]}, Type: {self.type}, Desc: '{self.description}', Imp: {self.importance})"


class Knowledge:
    """지식 베이스 데이터 구조"""

    def __init__(self, concept: str, description: str, embedding: list[float],
                last_updated: Optional[datetime.datetime] = None,
                 sources: Optional[List[str]] = None):
        self.id = str(uuid.uuid4())
        self.concept = concept
        self.description = description
        self.embedding = embedding
        self.timestamp = datetime.datetime.now()
        self.last_updated = last_updated or self.timestamp
        self.sources = sources or []

    def __repr__(self):
        return f"Knowledge(Concept: '{self.concept}', Desc: '{self.description}')"