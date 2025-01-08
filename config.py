import os

from yaml_kit import Config, get_and_run_config_command
from yaml_kit.config import Parameter


config = Config("StanceDetectionConfig")


@config.parameter("Experiment", types=str)
def name(val):
    """
    The name of this experiment. Used when generating output_dir.
    """
    assert val != ''


@config.parameter("Experiment", types=str, default="logs/")
def logdir(val):
    """
    The base directory for storing experiment logs.
    """
    assert val != ''


@config.parameter("Experiment", types=int, default=0)
def version(val):
    assert val >= 0


@config.parameter("Experiment", types=int, default=0)
def random_seed(val):
    pass


@config.parameter("Data", types=str)
def name(val):  # noqa
    """
    Name of the dataset to load. Get valid values using 
    python -m src.data.dataset
    """
    assert val != ''


@config.parameter("Data", types=str)
def datadir(val):
    """
    Path to directory containing in-target/ and images/ subdirectories.
    """
    assert os.path.isdir(val), "Data.datadir does not exist!"


@config.parameter("Data", types=str, default="none", deprecated=True)
def language(val):
    assert val == "none" or val != ''


@config.parameter("Data", types=str, default="en")
def prompt_language(val):
    """
    ISO code for the prompt language to use.
    """
    assert val != ''


@config.parameter("Data", types=str, default="en")
def tweet_language(val):
    """
    ISO code for the tweet language to use.
    """
    assert val != ''


@config.parameter("Data", types=bool, default=True)
def use_images(val):
    """
    If False, loads Gaussian noise.
    """
    pass


@config.parameter("Data", types=bool, default=False)
def use_image_text(val):
    """
    If True, don't load images, just text extracted from the images
    which are held in Data.images_dirname.
    """
    pass


@config.parameter("Data", types=str, default="images")
def images_dirname(val):
    """
    Name of subdirectory under datadir that contains the images to load.
    """
    assert val != ''


@config.parameter("Data", types=bool, default=True)
def use_text(val):
    """
    If False, loads the prompt without the text.
    """
    pass


@config.parameter("Model", types=str, default="qwen2")
def name(val):  # noqa
    """
    Name of the model family to load. Get valid values using 
    python -m src.modeling.model
    """
    assert val != ''


@config.parameter("Model", types=str)
def model_path(val):
    """
    The huggingface model path to load.
    """
    assert val != ''


@config.on_load
def set_output_dir():
    logdir = config.Experiment.logdir.value
    logdir = os.path.abspath(logdir)
    exp_name = config.Experiment.name.value
    model_name = config.Model.model_path.value
    model_name = model_name.split('/')[-1]
    prompt_lang = config.Data.prompt_language.value
    tweet_lang = config.Data.tweet_language.value
    version = config.Experiment.version.value
    seed = config.Experiment.random_seed.value
    version_str = f"version_{version}/seed_{seed}"
    outdir = os.path.join(logdir, exp_name, model_name,
                          f"prompt_{prompt_lang}",
                          f"tweet_{tweet_lang}",
                          version_str)
    if "output_dir" not in config.Experiment:
        config.Experiment.add(Parameter(
            "output_dir", value=outdir, types=str,
            comment="Automatically generated"))
    else:
        config.update("output_dir", outdir, group="Experiment",
                      run_on_load=False)


@config.on_load
def set_language_from_deprecated():
    lang = config.Data.language.value
    if lang != "none":
        config.update("tweet_language", lang, group="Data", run_on_load=False)
    else:
        config.update("language", config.Data.tweet_language.value,
                      group="Data", run_on_load=False)


if __name__ == "__main__":
    get_and_run_config_command(config)
