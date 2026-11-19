"""Entry point: ``python -m ml_model_dojo``."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from ml_model_dojo.app import main
    sys.exit(main())
