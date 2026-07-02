import json
from dwitracts.glm import DwiTractsGlm
import dwitracts.plot as dwiplot

config_file = 'project/config_glm_limbic.json'

with open(config_file, 'r') as myfile:
    params = json.loads(myfile.read())

params_gen = params['general']

with open(params_gen['tracts_config_file'], 'r') as myfile:
    params['tracts'] = json.loads(myfile.read())

with open(params_gen['preproc_config_file'], 'r') as myfile:
    params['preproc'] = json.loads(myfile.read())

my_glm = DwiTractsGlm(params)
assert my_glm.initialize()

assert my_glm.fit_glms(clobber=True, verbose=True)
assert my_glm.extract_distance_traces(clobber=True, verbose=True)
assert my_glm.extract_distance_traces_rft1d(clobber=True, verbose=True, debug=True)
