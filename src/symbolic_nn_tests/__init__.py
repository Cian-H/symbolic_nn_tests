import ssl

import typer

from . import __main__

ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore


def main():
    typer.run(__main__.main)
