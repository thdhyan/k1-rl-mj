"""Import the K1 tasks (registers them), then hand over to mjlab's train/play CLIs."""
import mjlab_k1  # noqa: F401  (registers Mjlab-Velocity-*-Booster-K1)


def train() -> None:
  from mjlab.scripts.train import main
  main()


def play() -> None:
  from mjlab.scripts.play import main
  main()
