#!/bin/env python
import typer
from rich.console import Console

from django.conf import settings
from cardio.management.commands.analyze import app as analyze_app
from cardio.management.commands.classify import app as classify_app
from cardio.management.commands.features import app as features_app
from cardio.management.commands.renamer import app as renamer_app
from cardio.management.commands.files import app as files_app

import os

console = Console()

app = typer.Typer(pretty_exceptions_enable=console.is_terminal)
app.add_typer(renamer_app)
app.add_typer(features_app)
app.add_typer(analyze_app)
app.add_typer(classify_app)
app.add_typer(files_app)


@app.callback()
def main(
        debug_level: int = typer.Option(default=1, help="Log level"),
        seed: int = typer.Option(default=None, help="Random seed")
):
    settings.DEBUG_LEVEL = debug_level
    if seed is not None:
        settings.SEEDS = [seed]


if __name__ == "__main__":
    os.nice(10)
    app()
