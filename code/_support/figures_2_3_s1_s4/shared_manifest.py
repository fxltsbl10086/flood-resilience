from pathlib import Path
import hashlib
def signature(path):
    p=Path(path)
    return {'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
