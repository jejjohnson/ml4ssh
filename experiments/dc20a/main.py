from absl import app
from absl import flags
from ml_collections import config_flags
import train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("my_config")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_string("stage", "train", "the experimental stage")


def main(_):

    # TODO: write download stage

    # TODO: write preprocessing stage
    # perform image experiment
    if FLAGS.stage == "train":
        train.train(FLAGS.my_config, None, None)
    else:
        raise NotImplementedError(f"Exp Stage implemented: {FLAGS.stage}")

    # TODO: write train more script

    # TODO: write inference script

    # TODO: write metrics script

    # TODO: write interactive script


if __name__ == "__main__":
    app.run(main)
