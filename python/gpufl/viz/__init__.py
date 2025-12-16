try:
    from .visualizer import init, show, compare, get_data
    from .reader import read_df, read_events
    from .timeline import (
        plot_kernel_timeline,
        plot_scope_timeline,
        plot_host_timeline,
        plot_memory_timeline,
        plot_utilization_timeline
    )
except ImportError:
    # Fallback if pandas/matplotlib are missing
    def show(*args, **kwargs):
        print("Error: gpufl[viz] dependencies (pandas/matplotlib) not installed.")

    def init(*args, **kwargs):
        print("Error: gpufl[viz] dependencies (pandas/matplotlib) not installed.")

    compare = show