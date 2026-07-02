import json
import sys
from dwitracts.main import DwiTracts
import dwitracts.plot as plot

print("Debug: Beginning compute_tsa.py")

config_file = sys.argv[1] if len(sys.argv) > 1 else 'project/config_tracts_lc.json'

with open(config_file, 'r') as myfile:
    params = json.loads(myfile.read())

my_dwi = DwiTracts( params )

assert my_dwi.initialize()


# Compute average streamline orientations

my_dwi.compute_average_orientations( verbose = True, clobber=True )

# Compute tract-specific anisotropy

my_dwi.compute_tsa( verbose=True, clobber=True )
