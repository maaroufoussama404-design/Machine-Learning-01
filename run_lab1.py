import multiprocessing
import runpy

if __name__ == "__main__":
    multiprocessing.freeze_support()
    runpy.run_path("lab01/src/benchmark.py", run_name="__main__")
