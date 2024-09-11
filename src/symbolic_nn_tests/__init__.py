from . import __main__

import typer
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


def main():
    typer.run(__main__.main)
