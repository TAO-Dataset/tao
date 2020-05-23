import multiprocessing as mp
from collections.abc import Iterable


_PoolWithContext_context = None


def _PoolWithContext_init(initializer, init_args):
    global _PoolWithContext_context
    _PoolWithContext_context = {}
    if init_args is None:
        initializer(context=_PoolWithContext_context)
    else:
        initializer(init_args, context=_PoolWithContext_context)


def _PoolWithContext_run(args):
    task_fn, task_args = args
    return task_fn(task_args, context=_PoolWithContext_context)


class PoolWithContext:
    """Like multiprocessing.Pool, but pass output of initializer to map fn.

    Usage:
        def init(context):
            context['init_return'] = 'init'
        def run(args, context):
            return (context['init_return'], args)
        p = PoolWithContext(4, init)
        print(p.map(run, ['task1', 'task2', 'task3']))
        # [('init', 'task1'), ('init', 'task2'), ('init', 'task3')]
        # NOTE: GPUs may be in different order
    """
    def __init__(self, num_workers, initializer, initargs=None):
        self.pool = mp.Pool(
            num_workers,
            initializer=_PoolWithContext_init,
            initargs=(initializer, initargs))

    def map(self, task_fn, tasks):
        return self.pool.map(_PoolWithContext_run,
                             ((task_fn, task) for task in tasks))

    def close(self):
        self.pool.close()

    def imap_unordered(self, task_fn, tasks):
        return self.pool.imap_unordered(_PoolWithContext_run,
                                        ((task_fn, task) for task in tasks))


if __name__ == "__main__":
    def _test_init(context):
        context['init_return'] = 'hi'

    def _test_init_2(context):
        context['hello'] = 2

    def _test_run(args, context):
        return (args, context['init_return'])

    def _test_run_2(args, context):
        return (args, context)

    p = PoolWithContext(4, _test_init)
    p2 = PoolWithContext(4, _test_init_2)
    print(p.map(_test_run, ['task1', 'task2', 'task3']))
    print(p2.map(_test_run_2, ['task1', 'task2', 'task3']))
