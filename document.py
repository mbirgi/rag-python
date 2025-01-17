from dataclasses import dataclass
from typing import Optional

@dataclass
class Document:
    text: str
    metadata: Optional[dict] = None