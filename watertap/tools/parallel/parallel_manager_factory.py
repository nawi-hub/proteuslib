from pyomo.common.dependencies import attempt_import
from watertap.tools.parallel.mpi_parallel_manager import MPIParallelManager
from watertap.tools.parallel.concurrent_futures_parallel_manager import (
    ConcurrentFuturesParallelManager,
)
from watertap.tools.parallel.single_process_parallel_manager import (
    SingleProcessParallelManager,
)

mpi4py, mpi4py_available = attempt_import("mpi4py")


def create_parallel_manager(parallel_manager_class=None, number_of_subprocesses=1):
    """
    Create and return an instance of a ParallelManager, based on the libraries available in the
    runtime environment.

    Allows an optional python class to be passed in as parallel_manager_class. If so, this class
    is instantiated and returned rather than checking the local environment.
    """
    if parallel_manager_class is not None:
        return parallel_manager_class()

    if has_mpi_peer_processes():
        return MPIParallelManager(mpi4py)

    if should_fan_out(number_of_subprocesses):
        return ConcurrentFuturesParallelManager(number_of_subprocesses)

    return SingleProcessParallelManager()


def has_mpi_peer_processes():
    """
    Returns whether the process was run as part of an MPI group with > 1 processes.
    """
    return mpi4py_available and mpi4py.MPI.COMM_WORLD.Get_size() > 1


def should_fan_out(number_of_subprocesses):
    """
    Returns whether the manager should fan out the computation to subprocesses. This
    is mutually exclusive with the process running under MPI.
    """
    return number_of_subprocesses > 1
