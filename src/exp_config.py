"""
Python module to define the command line parameters and some configuration functions
"""

import argparse
from argparse import Namespace
import ipdb
from dataclasses import dataclass, field, fields
from typing import List, Dict, Tuple, Union, Optional, Callable
import torch
from wandb.sdk.wandb_config import Config as WandbConfig
from config_vars import PHM_TOOLS, PHM_FAIL_TYPES

@dataclass
class ExperimentConfig:
    """
    This dataclass contains all the experiment configuration parameters.
    Its attributes are set by reading the ones defines in the yaml file of
    the experiment, but we can also set some parameters with some default values
    """

    model_config_path: str = "config/ssm_config.yaml"
    use_wandb: bool = False
    # train test split params
    test_size: float = 0.2
    val_size: float = 0.1
    test_idx: List[int] = field(default_factory=lambda: [0])
    # transformer params
    transformer_type: int = 1
    feature_type: str = "phm"
    scaler: str = "standard"
    scaler_kwargs: Dict[str, float] = field(
        default_factory=lambda: {
            "low_limit": -1.0,
            "high_limit": 1.0,
        }
    )
    # approach parameters
    approach: str = "windowed"
    # quantile regression parameters
    quantile_reg: bool = True
    quantile_dist: str = "uniform"
    quantile_run: float = 0.25
    bounds: List[float] = field(default_factory=lambda: [0.1, 0.9])
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    # training parameters
    batch_size: int = 32
    sequence_length: int = 170
    stride: int = 1
    epochs: int = 100
    lr: float = 1.0e-03
    weight_decay: float = 1.0e-04
    # loss parameters
    loss: str = "quantile_reg"
    eval_loss: str = "rmse"
    life_eval_loss: str = "rmse"
    # multi run parameters
    n_runs: int = 5
    start_run_id: int = 0
    run_id: int = 1
    # model summary
    summary_func: str = "torchinfo"
    # 500test_script parameters
    file_pos: int = 0
    # quantile plots
    full_life: bool = False
    ncols: int = 1
    nrows: int = 1
    life_idx: List[int] = field(default_factory=lambda: [3])
    col_idx: int = 0
    start_idx: int = 0
    end_idx: int = 1000000
    plot_run_id: int = 1
    quantile_plot: float = 0.25
    save_plot: bool = False
    show_plot: bool = False
    # raw signal plots
    use_test_set: bool = False
    # business metrics plots
    max_windows: int = 100
    n_maintenance_windows: int = 10
    # select_windows parameters
    max_rul: int = 500
    keep_long_rul_prob: float = 0.2
    n_const_win: int = 10
    normalize_rul: bool = False
    ad: bool = False
    # monotonic approach
    monotonic: bool = False
    # wandb sweep
    sweep_method: str = "random"
    sweep_param_names: List[str] = field(default_factory=lambda: ["batch_size", "lr"])
    batch_size_vals: List[int] = field(default_factory=lambda: [16, 32, 48])
    n_const_win_vals: List[int] = field(default_factory=lambda: [10, 100, 200])
    lr_vals: List[float] = field(default_factory=lambda: [1e-04, 1e-02])
    dropout_vals: List[float] = field(default_factory=lambda: [0.1, 0.2])
    sequence_length_vals: List[int] = field(default_factory=lambda: [10000, 20000])
    stride_vals: List[int] = field(default_factory=lambda: [100, 200])
    d_model_vals: List[float] = field(default_factory=lambda: [32, 64])

    @classmethod
    def from_dict(cls, config: dict) -> "ExperimentConfig":
        valid_keys = {f.name for f in fields(cls)}
        unknown = config.keys() - valid_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**config)

    def add_params(self, args: dict):
        """
        This function let's us to add some additional parameters
        to the dataclass (i.e. parameters passed through the command line)
        """
        for key in args:
            setattr(self, key, args[key])

# dataclass for the model configuration

@dataclass
class ModelConfig:
    """
    This dataclass contains all the configuration parameters for the
    models used in the project
    """

    random_init: bool = True
    d_model: int = 128
    n_layers: int = 5
    dropout: float = 0.0
    dropout_fn: Optional[Callable] = field(default=torch.nn.modules.dropout.Dropout1d)
    gap: bool = True
    # quantile regression
    quantile_reg: bool = True
    tau_mult: bool = True
    tau_feat: bool = True
    # monotonic
    n_mono_layers: int = 2
    n_neurons: int = 128
    n_mono_neurons: int = 128
    # S4 config
    lr: float = 1.0e-3
    activation: str = "relu"
    gate_act: str = "null"
    mult_act: str = "null"
    final_act: str = "glu"
    prenorm: bool = False
    # S4D config
    d_state: int = 64
    act: str = "gelu"
    # S5 config
    bidir: bool = False
    ff_dropout: float = 0.0
    attn_dropout: float = 0.0
    # RULTransformer config
    d_ff: int = 64
    n_heads: int = 8
    # RULInformer config
    factor: int = 5
    attn: str = "prob"
    inf_activation: str = "gelu"
    distil: bool = True
    output_attention: bool = False
    device: str = "cpu"

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        valid_keys = {f.name for f in fields(cls)}
        unknown = config.keys() - valid_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**config)

    def add_params(self, args: dict):
        """
        This function let's us to add some additional parameters
        to the dataclass (i.e. parameters passed through the command line)
        """
        for key in args:
            setattr(self, key, args[key])

def define_arguments() -> Namespace:
    """
    This function defines an ArgumentParser object to define the command line arguments
    and returns a Namespace object with the parsed arguments

    Args:
        No input arguments needed

    Returns:
        args (Namespace): namespace object with parsed arguments
    """

    parser = argparse.ArgumentParser(description="Experiment configuration command line parameters")

    parser.add_argument(
        "--exp_config_path",
        type=str,
        default="config/exp_config.yaml",
        help="Path to the experiment config file",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp",
        help="Experiment name",
    )

    parser.add_argument(
        "--sweep_name",
        type=str,
        default="sweep",
        help="wandb sweep name",
    )

    parser.add_argument(
        "--sweep_runs",
        type=int,
        default=10,
        help="number of hyperparameters configurations to try in a wandb sweep",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="ssm-pdm",
        help="Name of the wandb project",
    )

    parser.add_argument(
        "--data_name",
        type=str,
        default="PHM",
        help="Name of the dataset to use",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="S4",
        help="Name of the RUL prediction model to use",
    )

    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["S4"],
        help="Names of the models to insert in the final paper metrics table",
    )

    parser.add_argument(
        "--baseline_model_names",
        type=str,
        nargs="+",
        default=["mean"],
        help="Names of the baseline models",
    )

    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="+",
        default=["exp"],
        help="Names of the experiments for the different models for the multi_plot script",
    )

    parser.add_argument(
        "--plot_approach",
        type=str,
        default="padding_standard",
        help="Appproach to use for the multi_plot script",
    )

    parser.add_argument(
        "--failure_type",
        type=str,
        default="flow_low",
        help="Type of failure to consider",
    )

    parser.add_argument(
        "--train_phm_tools",
        type=str,
        nargs="+",
        default=["01M01"],
        help="Name of the ion milling machines to use in the training set",
    )

    parser.add_argument(
        "--test_phm_tools",
        type=str,
        nargs="+",
        default=["01M02"],
        help="Name of the ion milling machines to use in the test set",
    )

    parser.add_argument(
        "--cmapss_models",
        type=str,
        default="FD001",
        help="Name of the CMAPSS model to use",
    )

    parser.add_argument(
        "--cmapss_val_idx",
        type=int,
        nargs="+",
        default=[0,50],
        help="Range of indexes to use for the validation set",
    )

    parser.add_argument(
        "--cmapss_test_idx",
        type=int,
        nargs="+",
        default=[50,100],
        help="Range of indexes to use for the test set",
    )

    parser.add_argument(
        "--device_num",
        type=int,
        default=0,
        help="CUDA device number",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If set, use wandb to track the experiment"
    )

    parser.add_argument(
        "--test_script",
        action="store_true",
        help="If set, use the trainig script in evaluation mode"
    )

    parser.add_argument(
        "--save_best_model",
        action="store_true",
        help="If set, save the best model"
    )

    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="If set, save the predictions into a file"
    )

    parser.add_argument(
        "--save_combined_outputs",
        action="store_true",
        help="If set, save the combined predictions into a file"
    )

    parser.add_argument(
        "--save_outputs_quantile",
        action="store_true",
        help="If set, save the quantile regression predictions into a file"
    )
    parser.add_argument(
        "--save_outputs_run",
        action="store_true",
        help="If set, save the predictions into a file for the different runs"
    )

    parser.add_argument(
        "--return_outputs",
        action="store_true",
        help="If set, return the outputs"
    )

    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="If set, save the df with the average metrics over the lifes"
    )

    parser.add_argument(
        "--compute_business_metrics",
        action="store_true",
        help="If set, save the df with the average business metrics over the lifes"
    )

    parser.add_argument(
        "--save_mean_metrics_df",
        action="store_true",
        help="If set, save the df with the average metrics over the lifes"
    )

    parser.add_argument(
        "--save_metrics_df",
        action="store_true",
        help="If set, save the df with the average metrics over the lifes"
    )

    parser.add_argument(
        "--model_summary",
        action="store_true",
        help="If set, compute the model summary"
    )

    parser.add_argument(
        "--model_summary_manual",
        action="store_true",
        help="If set, compute the model summary manually"
    )

    parser.add_argument(
        "--save_summary_dict",
        action="store_true",
        help="If set, compute the model summary manually"
    )

    parser.add_argument(
        "--print_summary_metrics",
        action="store_true",
        help="If set, print the summary metrics"
    )

    parser.add_argument(
        "--print_mean_metrics_df",
        action="store_true",
        help="If set, print the mean metrics dataframe"
    )

    parser.add_argument(
        "--save_mean_metrics_df_runs",
        action="store_true",
        help="If set, save the df with the average metrics for the different runs"
    )

    parser.add_argument(
        "--sub_lifes_metrics",
        action="store_true",
        help="If set, compute the sub lifes metrics"
    )

    parser.add_argument(
        "--obsidian_table",
        action="store_true",
        help="If set, produce a markdown table"
    )

    parser.add_argument(
        "--continue_old_sweep",
        action="store_true",
        help="If set, continue an already started sweep"
    )

    parser.add_argument(
        "--get_test_idx",
        action="store_true",
        help="If set, get the indexes of the test lifes. Needed to produce the quantile plots"
    )

    parser.add_argument(
        "--sweep_id",
        type = str,
        default = "hxol4h28",
        help = "id of a wandb sweep"
    )

    parser.add_argument(
        "--wandb_entity",
        type = str,
        default = "frizzo-davide-Univeristy of Padova",
        help = "entity name of wandb"
    )

    parser.add_argument(
        "--an_score_plots",
        action="store_true",
        help="If set, produce the anomaly score plots"
    )

    parser.add_argument(
        "--combined_signal_plots",
        action="store_true",
        help="If set, produce the combined signals plots"
    )

    parser.add_argument(
        "--plot_rul",
        action="store_true",
        help="If set, produce the RUL signal in plot_raw_signals.py"
    )

    args = parser.parse_args()

    return args

def check_arguments(args: Union[ExperimentConfig, WandbConfig]) -> None:
    """
    This functions checks the validity of the experiment configuration through some assert statements

    Args:
        args (ExperimentConfig): experiment configuration object

    Returns:
        None: the function does not return nothing but can throw some exception if the arguments are not passed correctly
    """

    assert args.data_name == "PHM", "This function works just with the PHM dataset"
    assert set(args.train_phm_tools).issubset(PHM_TOOLS), (
        f"The set of train tools must be a subset of {PHM_TOOLS} but got {args.train_phm_tools}"
    )
    assert set(args.test_phm_tools).issubset(PHM_TOOLS), (
        f"The set of test tools must be a subset of {PHM_TOOLS} but got {args.test_phm_tools}"
    )
    assert args.failure_type in PHM_FAIL_TYPES, (
        f"Failure type name must be in {PHM_FAIL_TYPES} but got {args.failure_type}"
    )

    if hasattr(args, "stride"):
        assert args.stride <= args.sequence_length, f"The stride must be less or equal to the sequence length but got stride={args.stride} and sequence_length={args.sequence_length}"

    if "windowed" in args.approach:
        assert ("window" in args.loss) or ("mse" in args.loss), f"We are in a window based approach but I got loss {args.loss} which is not supported for this approach"

    if args.quantile_reg:
        assert ("quantile" in args.loss) and ("pinball" in args.eval_loss), f"You are using a quantile regression approach and loss is {args.loss} and eval_loss {args.eval_loss}. One of the two (or both) are not supported for this approach"


def set_exp_name(config: ExperimentConfig) -> str:
    """
    This function is used to set the experiment name (that will also be used
    to set the name of the wandb runs) based on the experiment configuration

    Args:
        config (ExperimentConfig): experiment configuration

    Returns:
        exp_name (str): string containing the experiment name
    """

    exp_name = f"super_benchmark_{config.model_name}_{config.failure_type}_{config.approach}"

    return exp_name

def set_sweep_name(config: ExperimentConfig) -> str:
    """
    This function is used to set the name of a wandb sweep 
    based on the sweep configuration

    Args:
        config (ExperimentConfig): experiment configuration

    Returns:
        sweep_name (str): string containing the sweep name
    """

    sweep_name = "ad_sweep" if config.ad else "sweep"
    sweep_name = f"{sweep_name}_{config.model_name}_{config.failure_type}_{config.approach}_{config.sweep_method}"

    for param_name in config.sweep_param_names:
        sweep_name = f"{sweep_name}_{param_name}"

    return sweep_name

def setup_exp() -> Tuple[ExperimentConfig, ModelConfig, torch.device, str]:
    """
    This functions sets up an experiment script defining the experiment configuration,
    model configuration, CUDA device and experiment name

    Args:
        No arguments required becuase the input arguments are passed through the command line
        or through the yaml files

    Returns:
        exp_config (ExperimentConfig): experiment configuration object
        model_config (ModelConfig): model configuration object
        device (torch.device): CUDA device to use for the GPU computations
        exp_name (str): experiment name
    """

    from utils import load_yaml_to_dict

    args = define_arguments()

    exp_config = load_yaml_to_dict(args.exp_config_path)
    exp_config = ExperimentConfig.from_dict(exp_config)
    exp_config.add_params(args=args.__dict__)

    check_arguments(args = exp_config)

    model_config = load_yaml_to_dict(exp_config.model_config_path)
    model_config = ModelConfig.from_dict(model_config)

    device = f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu"
    model_config.device = device

    print("-" * 50)
    print(f"Using device: {device}")
    print("-" * 50)

    if args.exp_name == "exp":

        exp_name = set_exp_name(exp_config)

        print("-" * 50)
        print(f"Experiment name set to {exp_name}")
        print("-" * 50)

    else:

        exp_name = args.exp_name

        print("-" * 50)
        print(f"Experiment name set to {exp_name}")
        print("-" * 50)

    return exp_config, model_config, device, exp_name
