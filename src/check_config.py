"""Small check so I can see step 1 working from the terminal.

Run from Task[03]:
    python -m src.check_config
"""

from src.config import ROOT, load_config, resolve_path


def main() -> None:
    cfg = load_config()
    print("project folder:", ROOT.name)
    print("project name:", cfg["project"]["name"])
    print("label:", cfg["problem"]["label"])
    print("metric:", cfg["problem"]["metric"])
    print("model file setting:", cfg["model"]["model_file"])
    print("model path resolves to:", resolve_path(cfg["model"]["model_file"]))


if __name__ == "__main__":
    main()
