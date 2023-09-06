class DefaultOptionsAsymptotic:
    maxiter = 30
    maxgap = 1e-6
    linesearchprecision = 1e-20
    linesearchminstep = 1e-3
    linearconstrainttolerance = 1e-10
    initmethod = 1
    verbose = 'no'
    maxgap_criteria=1
    option_names = [
        'maxiter',
        'maxgap',
        'linesearchprecision',
        'linesearchminstep',
        'linearconstrainttolerance',
        'initmethod',
        'verbose',
        'maxgap_criteria',
    ]


class DefaultOptionsAsymptoticStep2:
    epsilon = 0
    epsilonprime = 1e-12
    option_names = [
        'epsilon',
        'epsilonprime'
    ]


def get_default_opt(solver='Asymptotic'):
    if solver=='Asymptotic':
            return DefaultOptionsAsymptotic()
    elif  solver=='Asymptotic2':
            return DefaultOptionsAsymptoticStep2()

