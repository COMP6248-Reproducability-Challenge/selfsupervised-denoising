"""Base command case for CLI.
"""

from argparse import _SubParsersAction
from abc import ABCMeta, abstractmethod
from overrides import EnforceOverrides
from typing import Dict


class Command(EnforceOverrides, metaclass=ABCMeta):
    """Base class for system commmands. `configure` method should be implemented
    to attach the required command(s) to the provided parser. The command behaviour
    should run using the ``execute`` command with the parsed arguments.
    """

    def __init__(self):
        self.device_manager = None

    @abstractmethod
    def configure(self, parser: _SubParsersAction):
        """Add the command and any subparsing to the provided parser.

        Args:
            parser (_SubParsersAction): A parser to add the command to.
        """
        pass

    @abstractmethod
    def execute(self, args: Dict):
        """Execute command behaviour with parsed arguments.

        Args:
            args (Dict): Dictionary of parsed arguments including those from any parent
                argument parsers.
        """
        pass

    @abstractmethod
    def cmd(self) -> str:
        """The string that executes the command.

        Returns:
            str: Command - case sensitive.
        """
        pass
