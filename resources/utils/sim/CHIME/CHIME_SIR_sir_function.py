### =============================================================================
### NOTATIONAL CONVENTIONS:
###   Comments starting with single hash - '#' - are "normal" comments
###   Comments starting with double hash - '##' - represent name corresopnding
###       named component in hand-developed GroMEt representation
###   Comments starting with triple_hash - '###' - represent comment about differences
###       from original CHIME sir.py:
###       https://github.com/CodeForPhilly/chime/blob/develop/src/penn_chime/model/sir.py
### =============================================================================



# ===============================================================================
#  SIR, Subroutine, P. Hein
#  Updates all disease states given the current state values
# -------------------------------------------------------------------------------
#     Input/Output Variables:
#     s           Current amount of individuals that are susceptible
#     i           Current amount of individuals that are infectious
#     r           Current amount of individuals that are recovered
#     beta        The rate of exposure of individuals to persons infected with COVID-19
#     gamma       Rate of recovery for infected individuals
#     n           Total population size
#
#     State Variables:
#     s_n         Update to the amount of individuals that are susceptible
#     i_n         Update to the amount of individuals that are infectious
#     r_n         Update to the amount of individuals that are recovered
#     scale       A scaling factor to compute updated disease variables
#
# -------------------------------------------------------------------------------
#  Called by:   sim_sir
#  Calls:       None
# ==============================================================================
def sir(s, i, r, beta, gamma, n):
    """
    The SIR model, one time step
    :param s: Current amount of individuals that are susceptible
    :param i: Current amount of individuals that are infectious
    :param r: Current amount of individuals that are recovered
    :param beta: The rate of exposure of individuals to persons infected with COVID-19
    :param gamma: Rate of recovery for infected individuals
    :param n: Total population size
    :return:
    """
    s_n = (-beta * s * i) + s  # Update to the amount of individuals that are susceptible ## sir_s_n_exp
    i_n = (beta * s * i - gamma * i) + i  # Update to the amount of individuals that are infectious ## sir_i_n_exp
    r_n = gamma * i + r  # Update to the amount of individuals that are recovered ## sir_r_n_exp

    scale = n / (s_n + i_n + r_n)  # A scaling factor to compute updated disease variables ## sir_scale_exp

    s = s_n * scale  ## sir_s_exp
    i = i_n * scale  ## sir_i_exp
    r = r_n * scale  ## sir_r_exp
    return s, i, r


def main():
    """
    implements generic CHIME configuration without hospitalization calculation
    initializes parameters and population, calculates policy, and runs dynamics
    :return:
    """
    ###


    # initial population
    s = 100
    i = 1
    r = 0
    n = s + i + r

    # calculate beta under policy STUB hard-coded value
    beta = 0.1
    gamma = 0.3
    # simulate dynamics (corresponding roughly to run_projection() )

    print('(s,i,r)=',sir(s, i, r, beta, gamma, n))  ## simsir_loop_1_1_call_sir_exp
    return sir(s, i, r, beta, gamma, n)  ## simsir_loop_1_1_call_sir_exp
    

if __name__ == '__main__':
    main()
