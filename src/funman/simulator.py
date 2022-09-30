from funman.config import Config


class SimConfig(Config):
    pass


class Simulator(object):

    # STUB for now this assumes:
    #   run_sim is a function that runs the sim
    #   query is a function that consumes the results of run_sim and returns a bool
    def query_simulator(run_sim, query, config: Config):
        return query(run_sim())
