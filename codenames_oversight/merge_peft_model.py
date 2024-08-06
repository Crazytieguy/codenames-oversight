import typer

from peft import AutoPeftModelForCausalLM  # type: ignore
from typing import Optional

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(model_path_in: str, model_path_out: Optional[str] = None):
    if model_path_out is None:
        model_path_out = f"{model_path_in}-merged"
    model = AutoPeftModelForCausalLM.from_pretrained(model_path_in, device_map={"": "cpu"})
    model_out = model.merge_and_unload()
    model_out.save_pretrained(model_path_out)


if __name__ == "__main__":
    app()
