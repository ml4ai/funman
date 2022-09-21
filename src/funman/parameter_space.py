
class ParameterSpace(object):

    def __init__(self) -> None:
        pass
    
    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> 'ParameterSpace':
        return ParameterSpace()

    # STUB intersect parameters spaces
    @staticmethod
    def intersect(ps1, ps2) -> 'ParameterSpace':
        return ParameterSpace()

    # STUB construct space where all parameters are equal
    @staticmethod
    def construct_all_equal(ps) -> 'ParameterSpace':
        return ParameterSpace()

    # STUB compare parameter spaces for equality
    @staticmethod
    def compare(ps1, ps2) -> bool:
        raise NotImplementedError()