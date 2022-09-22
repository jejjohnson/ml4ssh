from absl import app
from absl import flags
from ml_collections import config_flags
import train_v2 as train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("my_config")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_string("experiment", "image", "the experimental")


def main(_):

    print("here!")
    # perform image experiment
    train.train(FLAGS.my_config, None, None)

    # TODO: write train more script

    # TODO: write inference script

    # TODO: write metrics script

    # TODO: write interactive script


if __name__ == "__main__":
    app.run(main)
