"""Configuration classes for the Engine and Task."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Type, TypeVar

@dataclass
class SerializableDataClass:
    """A data class that can be serialized to and from a dictionary."""

    def to_dict(self) -> Dict:
        """Convert the data class to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[TypeVar("T")], d: Dict) -> TypeVar("T"):
        """Create a data class from a dictionary."""
        return cls(**d)
