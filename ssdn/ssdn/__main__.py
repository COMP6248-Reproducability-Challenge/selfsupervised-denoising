"""Main method for interacting with denoiser through CLI.
"""

import sys
import ssdn
import ssdn.cli

from typing import List


def start_cli(args: List[str] = None):
    ssdn.logging_helper.setup()
    if args is not None:
        sys.argv[1:] = args
    ssdn.cli.start()


if __name__ == "__main__":
    start_cli()
