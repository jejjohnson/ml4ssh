from absl import app
from absl import flags
from ml_collections import config_flags
import data
import train
import train_more

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("my_config")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_string("dldir", None, "data directory")
flags.DEFINE_string("stage", "train", "the experimental stage")


def main(_):

    if FLAGS.stage == "download":
        # TODO: write download stage
        data.download(FLAGS.dldir, dataset="obs")
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
        # TODO: write inference script
        raise NotImplementedError()

    elif FLAGS.stage == "metrics":
        # TODO: write metrics script
        raise NotImplementedError()

    elif FLAGS.stage == "figures":
        # TODO: write interactive script
        raise NotImplementedError()

    else:
        raise NotImplementedError(f"Exp Stage implemented: {FLAGS.stage}")


if __name__ == "__main__":
    app.run(main)
