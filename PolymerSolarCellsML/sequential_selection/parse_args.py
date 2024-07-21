from dataclasses import dataclass, field

from argument_parser import HfArgumentParser

@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    write_output: bool = field(
        default=False, metadata={"help": "Write the generated output to a json file"}
    )
    run_parallel_paths: bool = field(
        default=False, metadata={"help": "Run GP bandits and linear contextual bandits in parallel"}
    )
    filtering_criteria: str = field(
        default='median', metadata={"help": "Mode for filtering PCE values, use the median, or earliest, or smallest or largest PCE value"}
    )
    plotting_criteria: str = field(
        default='earliest', metadata={"help": "Mode for picking the simulated material history, use the median, or shortest, or longest path or earliest random seed"}
    )
    acceptor_type: str = field(
        default='both', metadata={"help": "Acceptor type to use for the dataset, either FA, NFA or both"}
    )
    random_seed: int = field(
        default=47, metadata={"help": "Random seed"}
    )
    alpha: float = field(
        default=1.0, metadata={"help": "Alpha value for the Thompson sampler"}
    )
    property_name: str = field(
        default='power conversion efficiency', metadata={"help": "Property to use for the sequential selection algorithm"}
    )
    gp_runs: int = field(
        default=10, metadata={"help": "Number of runs to use for Gaussian process selection algorithms"}
    )
    use_gpr_noise: bool = field(
        default=False, metadata={"help": "Use noise in the GPR model if True"}
    )
    kernel: str = field(
        default='RBF', metadata={"help": "Kernel to use for the GPR model. Options are Matern, RBF and RationalQuadratic"}
    )
    figures_dir: str = field(
        default='../../output/', metadata={"help": "Directory to save the figures in"}
    )
    
def parse_args(args):
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses(args=args)
    return args


