"""
Python script to put together the RUL predictions produced by the quantile_reg
mode for different quantile levels
"""

# general imports
import os
import sys
import ipdb
import numpy as np
import matplotlib.pyplot as plt

src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)

from utils import (
    generate_path,
    load_yaml_to_dict,
    ExperimentConfig,
    get_most_recent_file,
    get_most_recent_dir,
    get_current_time,
    open_element,
)

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

config_path=os.path.join(experiment_path,"config","ssm_exp_config.yaml")
config=load_yaml_to_dict(config_path)
config=ExperimentConfig(config)

# Set the plot path
plot_path = generate_path(basepath=experiment_path,
                          folders=["plots",
                                   config.model_name,
                                   config.cmapss_models,
                                   config.approach,
                                    "quantile_reg"])
plot_path = get_most_recent_dir(plot_path,file_pos=config.file_pos)

# Get the outputs directory of the most recent experiment
outputs_path = generate_path(basepath=experiment_path,
                                folders=[
                                    "outputs",
                                    config.model_name,
                                    config.cmapss_models,
                                    config.approach,
                                    "quantile_reg",
                                ])
outputs_dirpath = get_most_recent_dir(outputs_path,file_pos=config.file_pos)

quantile_preds = {}

if config.life_idx is None:
    config.life_idx = np.arange(config.nrows*config.ncols)
else:
    assert config.nrows*config.ncols == len(config.life_idx), "Number of rows and columns must match the number of lives"

for quantile in config.quantiles:
    quantile_path = generate_path(basepath=outputs_dirpath,
                                  folders=[f"quantile_{quantile}"])
    quantile_filepath = get_most_recent_file(quantile_path,file_pos=config.file_pos)
    ouputs_dict = open_element(quantile_filepath)
    y_pred,y_true = ouputs_dict["y_pred"],ouputs_dict["y_true"]

    if config.approach == "padding":
        pred,true = y_pred[config.life_idx,:],y_true[config.life_id,:]
    elif config.approach == "windowed":
        pred = [y_pred[i] for i in config.life_idx]
        true = [y_true[i] for i in config.life_idx]
    if not config.full_life:
        mask = true!=0 if config.approach=="padding" else [true[i]!=0 for i in range(len(true))]
    
    quantile_preds[f"quantile_{quantile}"] = pred

# Produce the plot
colors = ["orange","green","red","purple","brown","pink","gray","olive","cyan"]
fig, axs = plt.subplots(config.nrows,config.ncols,figsize=(30,20))
for i in range(config.nrows):
    for j in range(config.ncols):
        if i*config.ncols+j<(config.nrows*config.ncols):

            ax=axs[i,j]
            ax.plot(true[i*config.ncols+j][mask[i*config.ncols+j]],color="blue",label='True RUL')
            for k,quantile in enumerate(config.quantiles):
                ax.plot(
                    quantile_preds[f"quantile_{quantile}"][i*config.ncols+j][mask[i*config.ncols+j]],
                    color=colors[k],
                    label=f'Quantile {quantile}'
                )
            ax.set_title(f'Life {config.life_idx[i*config.ncols+j]+config.test_idx[0]+1}')
            ax.set_xticks([])
            ax.set_ylabel('RUL')
            ax.legend()

if config.save_plot:

    if config.full_life:
        filename=f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_quantile_reg_global_predictions_grid_full"
    else:
        filename=f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_quantile_reg_global_predictions_grid_pad"

    life_idx_str="_".join(str(x) for x in config.life_idx)
    filename=f"{filename}_life_{life_idx_str}.pdf"

    plot_path=os.path.join(plot_path,filename)
    plt.savefig(plot_path,bbox_inches='tight')
    print('#'*50)
    print(f'Plot saved at: {plot_path}')
    print('#'*50)
