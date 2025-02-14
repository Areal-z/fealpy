from . import _pangulu_r64_cpu

solver_classes = {
        'r64_cpu': _pangulu_r64_cpu.r64_cpu_solver,
        }

class PanguLUSolver:
    def __new__(cls, data_type='r', precision='64', platform='cpu', **kwargs):
        version = f"{data_type[0]}{precision}_{platform}"
        return solver_classes[version](**kwargs)
