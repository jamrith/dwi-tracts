
import json
from dwitracts.glm import DwiTractsGlm
import dwitracts.plot as dwiplot

# %%
# GLM Analysis using DwiTractsGlm class

for config_file in ['config_glm_tau.json', 'config_glm_moca.json',
                    'config_glm_tau_lc_ent.json', 'config_glm_tau_mt.json',
                    'config_glm_bihemi_tau_mt.json', 'config_glm_bihemi_tau.json']:

    config_file = 'project/'+config_file

    with open(config_file, 'r') as myfile:
        params = json.loads(myfile.read())
        
    params_gen = params['general']
    params_glm = params['glm']

    with open(params_gen['tracts_config_file'], 'r') as myfile:
        json_string=myfile.read()
        
    params['tracts'] = json.loads(json_string)

    with open(params_gen['preproc_config_file'], 'r') as myfile:
        json_string=myfile.read()
        
    params['preproc'] = json.loads(json_string)

    my_glm = DwiTractsGlm( params )

    # Initialize the object with the current parameters
    assert my_glm.initialize( )


    # %%
    # Fit all specified GLMs and save results to target directory
    assert my_glm.fit_glms( clobber=True, verbose=True )


    # %%
    # Extract distance traces from the GLM results
    assert my_glm.extract_distance_traces( clobber=True, verbose=True )


    # %%
    # Extract distance traces from the GLM results, using RFT1D cluster inference
    assert my_glm.extract_distance_traces_rft1d( clobber=True, verbose=True, debug=True )



    # %%
    # Extract distance traces from the GLM results, using cluster inference and permutation testing
    # my_glm.extract_distance_trace_clusters( clobber=True, verbose=True, debug=True )


    # %%
    # Plot t-value distance traces

    # Font sizes
    params['axis_font'] = 24
    params['yticklabel_font'] = 16
    params['xticklabel_font'] = 20
    params['title_font'] = 40
    params['font_scale'] = 1.5
    params['image_format'] = 'svg'
    params['alpha'] = 0.05
    params['write_data'] = True

    alpha = 0.05

    tract_names = my_glm.tract_names.copy()
    tract_names.sort()

    # Uncorrected
    params['stat_type'] = 'uncorrected'
    dwiplot.plot_distance_traces( params, tract_names )

    # RFT corrected
    params['stat_type'] = 'rft'
    dwiplot.plot_distance_traces( params, tract_names )

    # Permutation corrected
    # dwiplot.plot_distance_traces( params, tract_names, alpha=alpha, stat_type='perm' )


    # %%
    # Plot scatterplots & violin plots for GLM results

    # Specify GLM
    for glm in params_glm.keys():

        # Set up parameters
        params_plot = {}
        params_plot['glm'] = glm
        params_plot['alpha'] = 0.05
        params_plot['outlier_z'] = params_glm[glm]['outlier_z']
        params_plot['stat_type'] = 'rft'
        params_plot['axis_font'] = 40
        params_plot['marker_size'] = 15
        params_plot['line_width'] = 4
        params_plot['ticklabel_font'] = 33
        params_plot['title_font'] = 45
        params_plot['legend_font'] = 40
        params_plot['legend_scale'] = 3
        params_plot['show_title'] = True
        params_plot['show_legend'] = True
        params_plot['dimensions'] = (12,10)
        params_plot['font_scale'] = 0.6
        params_plot['write_csv'] = False
        params_plot['image_format'] = 'svg'
        params_plot['write_data'] = True

        # Loop through tracts and call plot function for each
        dwiplot.plot_glm_results_all( params_plot, my_glm, verbose=True )
        

    # %%
    # Aggregate tracts into single volumes
    assert my_glm.aggregate_tracts(prefix='tract_final_norm_bidir', op='max', verbose=True)

    # %%
    # Aggregate stats into single volumes

    assert my_glm.aggregate_stats( op='max', pthres=0.05, suffix='-rft', verbose=True, clobber=True )


    # %%
    # Generate Pajek graphs

    assert my_glm.create_pajek_graphs( suffix='-rft', edge_val='tsum', verbose=True, clobber=True )

