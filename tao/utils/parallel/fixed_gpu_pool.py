import multiprocessing as mp
from tao.utils.parallel.pool_context import PoolWithContext


class FixedGpuPool:
    """Pool where each process is attached to a specific GPU.

    Usage:
        def init(args, context):
            context['init_return'] = 'init'
        def run(args, context):
            return (context['gpu'], context['init_return'], args)
        p = FixedGpuPool([0, 1, 2, 3], init, None)
        print(p.map(run, ['task1', 'task2', 'task3']))
        # [(0, 'init', 'task1'), (1, 'init', 'task2'), (2, 'hi', 'task3')]
        # NOTE: GPUs may be in different order
    """

    def __init__(self, gpus, initializer=None, initargs=None):
        gpu_queue = mp.Manager().Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)
        self.pool = PoolWithContext(
            len(gpus), _FixedGpuPool_init, (gpu_queue, initializer, initargs))

    def map(self, task_fn, tasks):
        return self.pool.map(_FixedGpuPool_run,
                             ((task_fn, task) for task in tasks))

    def imap_unordered(self, task_fn, tasks):
        return self.pool.imap_unordered(_FixedGpuPool_run,
                                        ((task_fn, task) for task in tasks))

    def close(self):
        self.pool.close()


def _FixedGpuPool_init(args, context):
    gpu_queue, initializer, initargs = args
    context['gpu'] = gpu_queue.get()
    initializer(initargs, context=context)


def _FixedGpuPool_run(args, context):
    task_fn, task_args = args
    return task_fn(task_args, context=context)


if __name__ == "__main__":
    def _test_gpu_init(args, context):
        context['init_return'] = 'init'

    def _test_gpu_run(args, context):
        return (context['gpu'], context['init_return'], args)

    p = FixedGpuPool([0, 1, 2, 3], _test_gpu_init, 'init arg')
    print(p.map(_test_gpu_run, ['task1', 'task2', 'task3']))
