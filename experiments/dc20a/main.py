from absl import app
from absl import flags
from ml_collections import config_flags
import data
import train
import train_more
import metrics
import viz
import inference

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("my_config")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_string("dldir", None, "data directory")
flags.DEFINE_string("results_name", None, "directory + save name for results file")
flags.DEFINE_string("variable_name", "ssh", "variable name in xr dataset")
flags.DEFINE_string("figure", "density", "the figure type for viz")
flags.DEFINE_string("figure_dir", None, "the directory for the figures")
flags.DEFINE_string("stage", "train", "the experimental stage")


def main(_):

    if FLAGS.stage == "download_obs":
        data.download(FLAGS.dldir, dataset="obs")

    elif FLAGS.stage == "download_ref":
        data.download(FLAGS.dldir, dataset="ref")

    elif FLAGS.stage == "preprocess":
        data.preprocess(FLAGS.my_config)

    elif FLAGS.stage == "ml_ready":
        data.ml_ready(FLAGS.my_config, experiment="swot1nadir5")
        data.ml_ready(FLAGS.my_config, experiment="swot1nadir1")
        data.ml_ready(FLAGS.my_config, experiment="nadir1")
        data.ml_ready(FLAGS.my_config, experiment="nadir4")

    elif FLAGS.stage == "train":
        train.train(FLAGS.my_config, None, None)

    elif FLAGS.stage == "train_more":
        # TODO: write train more script
        train_more.train(FLAGS.my_config, None, None)

    elif FLAGS.stage == "inference":
        inference.inference(FLAGS.my_config, FLAGS.results_name)

    elif FLAGS.stage == "metrics":
        metrics.metrics(FLAGS.my_config, FLAGS.results_name, FLAGS.variable_name)

    elif FLAGS.stage == "viz":
        # TODO: write interactive script
        viz.viz(
            config=FLAGS.my_config,
            figure=FLAGS.figure,
            resultsfile=FLAGS.results_name,
            savedir=FLAGS.figure_dir,
            variable_name=FLAGS.variable_name,
        )
    else:
        raise NotImplementedError(f"Exp Stage implemented: {FLAGS.stage}")


if __name__ == "__main__":
    app.run(main)
