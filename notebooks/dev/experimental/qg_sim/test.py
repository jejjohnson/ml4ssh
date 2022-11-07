from absl import app
from absl import flags

from ml_collections import config_flags

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("my_config")
_MY_FLAG = flags.DEFINE_integer("my_flag", 1, "flagggg")
_MY_FLAG2 = flags.DEFINE_integer("my_flag2", 10, "flagggg2")


def main(_):
    print(_CONFIG.value)
    print(_MY_FLAG.value)
    print(_MY_FLAG2.value)


if __name__ == "__main__":
    app.run(main)
