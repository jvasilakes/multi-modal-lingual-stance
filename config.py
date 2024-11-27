import os

from experiment_config import Config, get_and_run_config_command
from experiment_config.config import Parameter


config = Config("StanceDetectionConfig")


@config.parameter("Experiment", types=str)
def name(val):
    assert val != ''


@config.parameter("Experiment", types=str, default="logs/")
def logdir(val):
    assert val != ''


@config.parameter("Experiment", types=int, default=0)
def version(val):
    assert val >= 0


@config.parameter("Experiment", types=int, default=0)
def random_seed(val):
    pass


@config.parameter("Data", types=str)
def name(val):  # noqa
    assert val != ''


@config.parameter("Data", types=str)
def datadir(val):
    assert os.path.isdir(val), "Data.datadir does not exist!"


@config.parameter("Model", types=str)
def name(val):  # noqa
    assert val != ''


@config.on_load
def set_output_dir():
    logdir = config.Experiment.logdir.value
    logdir = os.path.abspath(logdir)
    exp_name = config.Experiment.name.value
    model_name = config.Model.name.value
    version = config.Experiment.version.value
    seed = config.Experiment.random_seed.value
    version_str = f"version_{version}/seed_{seed}"
    outdir = os.path.join(logdir, exp_name, model_name, version_str)
    if "output_dir" not in config.Experiment:
        config.Experiment.add(Parameter(
            "output_dir", value=outdir, types=str,
            comment="Automatically generated"))
    else:
        config.update("output_dir", outdir, group="Experiment",
                      run_on_load=False)


if __name__ == "__main__":
    get_and_run_config_command(config)
