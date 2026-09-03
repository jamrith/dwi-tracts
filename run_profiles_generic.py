#!/usr/bin/env python3
"""
run_profiles_generic.py <config.json> [--no-inference] [--inference-only]
                                        [--export-only]

Config-driven driver for the along-tract profile GLM (dwitracts.profiles,
PROFILE_GLM_SPEC.md). Takes the SAME config a run_glm_generic.py run takes,
plus a "profiles" block -- see DwiTractsProfiles' docstring for the schema.

The legacy path is untouched: this writes under profiles.output_dir, so the
same config can be run both ways and the two trees compared directly (the
comparison plan in PROFILE_GLM_SPEC.md section 8).

    python run_glm_generic.py      project/config_glm_x.json   # baseline
    python run_profiles_generic.py project/config_glm_x.json   # new path
"""
import sys
import json

from dwitracts.profiles import DwiTractsProfiles


def load_params(config_file):
    with open(config_file, 'r') as f:
        params = json.loads(f.read())

    params_gen = params['general']
    with open(params_gen['tracts_config_file'], 'r') as f:
        params['tracts'] = json.loads(f.read())
    with open(params_gen['preproc_config_file'], 'r') as f:
        params['preproc'] = json.loads(f.read())

    # Same override run_glm_generic.py applies: run against a standard DTI
    # metric (FA/MD/RD/AD) instead of the default TSA regression beta.
    if 'metric' in params_gen:
        params['tracts']['dwi_regressions']['metric_stem'] = params_gen['metric']

    return params


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    config_file = sys.argv[1]
    flags = set(sys.argv[2:])

    params = load_params(config_file)
    if 'profiles' not in params:
        print('Config has no "profiles" block; nothing to do. See '
              'PROFILE_GLM_SPEC.md section 7 / DwiTractsProfiles docstring.')
        return 1

    prof = DwiTractsProfiles(params)
    assert prof.initialize()

    if '--export-only' in flags:
        # Regenerate shell_geometry.csv / inference_meta-*.csv for a tree that
        # already exists, without repeating any fit or permutation.
        assert prof.export_metadata(verbose=True)
    else:
        if '--inference-only' not in flags:
            assert prof.compute_profiles(clobber=True, verbose=True)
        if '--no-inference' not in flags:
            assert prof.run_inference(clobber=True, verbose=True)

    print('DONE PROFILES:', config_file, '->', prof.profiles_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
