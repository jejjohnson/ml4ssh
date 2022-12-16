from absl import app
from absl import flags
from ml_collections import config_flags

# import train_v2 as train
import data

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("my_config")
flags.DEFINE_string("stage", "train", "the experimental stage")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_string("datadir", None, "data download directory")
flags.DEFINE_string("credentials", None, "credentials file directory")
flags.DEFINE_string("experiment", "image", "the experimental")


def main(_):

    # =============================
    # DOWNLOAD DATA
    # =============================
    if FLAGS.stage == "download_obs":
        data.download(FLAGS.datadir, FLAGS.credentials, "obs")

    elif FLAGS.stage == "download_correction":
        data.download(FLAGS.datadir, FLAGS.credentials, "correction")

    elif FLAGS.stage == "download_results":
        data.download(FLAGS.datadir, FLAGS.credentials, "results")

    # =============================
    # PREPROCESS DATA
    # =============================
    elif FLAGS.stage == "preprocess":
        raise NotImplementedError()

    elif FLAGS.stage == "ml_ready":
        data.ml_ready(FLAGS.my_config)

    elif FLAGS.stage == "train":
        # train.train(FLAGS.my_config, None, None)
        raise NotImplementedError()

    elif FLAGS.stage == "train_more":
        raise NotImplementedError()

    elif FLAGS.stage == "inference":
        raise NotImplementedError()

    elif FLAGS.stage == "metrics":
        raise NotImplementedError()

    elif FLAGS.stage == "viz":
        raise NotImplemented()


if __name__ == "__main__":
    app.run(main)
