import sys

import numpy as np
import pandas as pd
import sklearn

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)