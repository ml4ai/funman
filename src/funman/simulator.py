
# STUB for now this assumes:
#   run_sim is a function that runs the sim
#   query is a function that consumes the results of run_sim and returns a bool
def query_simulator(run_sim, query):
    return query(run_sim())