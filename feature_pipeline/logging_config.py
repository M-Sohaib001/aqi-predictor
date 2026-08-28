"""
One place to configure logging. Using the `logging` module instead of
`print()` everywhere means: log levels (so CI output isn't noisy),
consistent formatting with timestamps, and no risk of a stray `print(data)`
accidentally dumping a secret-bearing API response into GitHub Actions logs.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )