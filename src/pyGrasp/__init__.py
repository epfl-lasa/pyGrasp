import os
import sys
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))

# Add path to find the cat module
cat_path = Path(here).parent.parent / Path("external/src/Neural-JSDF")
sys.path.append(str(cat_path.absolute()))
