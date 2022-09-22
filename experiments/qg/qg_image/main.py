from absl import app
from absl import flags
from ml_collections import config_flags
import train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("my_config")
flags.DEFINE_string("workdir", None, "work directory")


def main(_):

    # perform training
    train.train(FLAGS.my_config, FLAGS.workdir, None)

    # TODO: write train more script

    # TODO: write inference script

    # TODO: write metrics script

    # TODO: write interactive script


if __name__ == "__main__":
    app.run(main)
