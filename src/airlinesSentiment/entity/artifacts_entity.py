from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataTransformationArtifact:
    vectorizer_path: Path
    cleaned_data_path: Path
    features_path: Path
    