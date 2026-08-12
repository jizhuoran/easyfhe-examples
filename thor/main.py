"""CLI assembly for the canonical u64 THOR example."""

from .benchmark import run_benchmark, validate_benchmark_request
from .config import parse_args
from .reference import load_reference_assets
from .runtime import create_runtime


def main(argv=None):
    config = parse_args(argv)
    assets = load_reference_assets(config)
    validate_benchmark_request(config, assets)
    client, runtime = create_runtime(config.weights_path)
    return run_benchmark(client, runtime, assets, config)


if __name__ == "__main__":
    main()
