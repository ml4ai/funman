import sympy
from sympy import exp, series, symbols

### Example from Scenario 1(a)
I, S, kappa, beta_c, beta_s, k, t, t_0, N = symbols(
    "I S kappa beta_c beta_s k t t_0 N"
)
eqn_sample = (
    I
    * S
    * kappa
    * (beta_c + (-beta_c + beta_s) / (1 + exp(-k * (-t + t_0))))
    / N
)

### Substitute numerical values for variables other than t
eqn_sample_sub = eqn_sample.subs(
    [
        (S, 5599999),
        (I, 1),
        (N, 5600000),
        (beta_c, 0.4),
        (beta_s, 1),
        (k, 5),
        (kappa, 0.45454545454545453),
        (t_0, 89),
    ]
)

### Substitute numerical values for variables other than t, t0
eqn_sample_sub_partial = eqn_sample.subs(
    [
        (S, 5599999),
        (I, 1),
        (N, 5600000),
        (beta_c, 0.4),
        (beta_s, 1),
        (k, 5),
        (kappa, 0.45454545454545453),
    ]
)

### Series expansions: : works well when t is left as a free variable and the series is expanded in terms of t
# eqn_sample_sub_series = sympy.series(eqn_sample_sub, t)

### Printing terms in series expansion
# for elt in eqn_sample_sub_series.args:
#     print(elt)


### Continued example from 1(a), but leaving both t and t_0 as free variables and doing series expansions in terms of both
eqn_sample_sub_series_partial = sympy.series(eqn_sample_sub_partial, t)
# eqn_sample_sub_series_partial = sympy.series(eqn_sample_sub_series_partial, t_0)

for elt in eqn_sample_sub_series_partial.args:
    # print(elt)
    print(sympy.series(elt, t_0))
