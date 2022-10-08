from funman.search_episode import SearchEpisode


class ParameterSpace(object):
    def __init__(self, episode: SearchEpisode) -> None:
        self.true_boxes = episode.true_boxes
        self.false_boxes = episode.false_boxes

    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB intersect parameters spaces
    @staticmethod
    def intersect(ps1, ps2) -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB construct space where all parameters are equal
    @staticmethod
    def construct_all_equal(ps) -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB compare parameter spaces for equality
    @staticmethod
    def compare(ps1, ps2) -> bool:
        raise NotImplementedError()
        raise NotImplementedError()
