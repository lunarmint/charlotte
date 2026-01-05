from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def charlotte(output: Annotated[str, typer.Option("--output", "-o")]) -> None:
    with open(output, "r", encoding="utf-8") as file:
        read_data = file.read()
        print(read_data)


if __name__ == "__main__":
    app()
