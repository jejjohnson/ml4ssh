import datetime
from datetime import timezone
from typing import Optional


def get_current_timestamp(tz: Optional[timezone] = timezone.utc) -> str:
    """This gives the current time stamp in a readable format.

    Args:
        tz (str, optional): The time zone. Defaults to timezone.utc.

    Returns:
        str: the current time stamp in date format
    """
    # get current time stamp
    time_stamp = datetime.datetime.now().timestamp()

    # convert to legible datetime
    time_stamp = datetime.datetime.fromtimestamp(time_stamp, tz=tz)

    return str(time_stamp)
