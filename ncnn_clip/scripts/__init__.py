import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).parent.parent
print(MODULE_ROOT)
sys.path.append(str(MODULE_ROOT))
