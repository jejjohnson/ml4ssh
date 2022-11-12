from absl import app
from absl import flags
from ml_collections import config_flags
import train
import train_more

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("my_config")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_string("stage", "train", "the experimental stage")


def main(_):

    if FLAGS.stage == "download":
        # TODO: write download stage
        raise NotImplementedError()

    elif FLAGS.stage == "preprocess":
        # TODO: write preprocessing stage
        raise NotImplementedError()

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
