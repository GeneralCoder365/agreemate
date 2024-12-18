# run_experiments.py
"""
Main entry point for running AgreeMate negotiation experiments.
Handles command line interface, logging setup, and experiment execution.
"""
import sys, logging, asyncio, argparse
from pathlib import Path
from datetime import datetime

from experiment_runner import ExperimentRunner
from config import EXPERIMENT_CONFIGS, validate_config

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create and setup experiment output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"

    # create directory structure
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    # setup logging to file
    log_path = exp_dir / "logs" / "experiment.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    return exp_dir


async def run_experiment(args):
    """Run experiment with given configuration."""
    # validate configuration
    if args.config not in EXPERIMENT_CONFIGS:
        available = ", ".join(EXPERIMENT_CONFIGS.keys())
        logger.error(f"Unknown configuration '{args.config}'. Available: {available}")
        sys.exit(1)

    config = EXPERIMENT_CONFIGS[args.config]
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Invalid configuration: {str(e)}")
        sys.exit(1)

    # setup experiment directory
    exp_name = args.name or args.config
    exp_dir = setup_experiment_dir(args.output, exp_name)
    logger.info(f"Running experiment '{exp_name}' in {exp_dir}")

    try:
        # initialize and run experiment
        runner = ExperimentRunner(
            config_name=args.config,
            output_dir=str(exp_dir),
            experiment_name=exp_name
        )

        results = await runner.run()

        # log completion summary
        logger.info("Experiment completed successfully:")
        logger.info(f"Total scenarios: {results['results'].scenarios_total}")
        logger.info(f"Completed: {results['results'].scenarios_completed}")
        logger.info(f"Failed: {results['results'].scenarios_failed}")
        logger.info(f"Results saved to: {exp_dir / 'results'}")

        return results

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AgreeMate negotiation experiments"
    )

    # required arguments
    parser.add_argument(
        "--output",
        required=True,
        help="Base directory for experiment outputs"
    )

    # optional arguments
    parser.add_argument(
        "--config",
        default="baseline",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment configuration to use"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Custom name for this experiment run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # run experiment
    try:
        asyncio.run(run_experiment(args))
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()