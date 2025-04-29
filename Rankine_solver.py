import numpy as np
from scipy.optimize import root_scalar
from apdx_functions import *

# Rankine cycle solver
def known(var):
    """
    Checks if a variable is defined (not NaN).
    
    Parameters:
    var (float): The variable to check.
    
    Returns:
    bool: True if the variable is defined, False otherwise.
    """
    return not np.isnan(var)

def unknown(var):
    """
    Checks if a variable is undefined (NaN).
    
    Parameters:
    var (float): The variable to check.
    
    Returns:
    bool: True if the variable is undefined, False otherwise.
    """
    return np.isnan(var)

def define_empty_variables():
    """
    Initializes the variables for the Rankine cycle simulation.
    
    Returns:
    dict: Updated main dictionary with calculated variables.
    """
    # Initialize all variables to NaN
    variables = {
        'm_dot':np.nan, # Mass flow rate (kg/s)
        'Qh_dot': np.nan,  # Heat addition rate (kW or kJ/s)
        'qh': np.nan,  # Heat addition (kJ/kg)
        'Qc_dot': np.nan,  # Heat rejection rate (kW or kJ/s)
        'qc': np.nan,  # Heat rejection (kJ/kg)
        'Wc_dot': np.nan,  # Work done by the compressor (kW or kJ/s)
        'wc': np.nan,  # Work done by the compressor (kJ/kg)
        'n':  np.nan,  # Efficiency (dimensionless)
        'COP_hp': np.nan, # Coefficient of performance (dimensionless)

        '1':{
            'T': np.nan,  # °C
            'P': np.nan,  # MPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan,  # kJ/kg
            'x': np.nan  # Quality (dimensionless)
        },

        '2':{
            'T': np.nan,  # °C
            'P': np.nan,  # MPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan,  # kJ/kg
            'x': 1  # Quality (dimensionless)
        },

        '3':{
            'T': np.nan,  # °C
            'P': np.nan,  # MPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan  # kJ/kg
        },

        '3b':{
            'T': np.nan,  # °C
            'P': np.nan,  # MPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan,  # kJ/kg
            'x': 1  # Quality (dimensionless)
        },

        '4':{
            'T': np.nan,  # °C
            'P': np.nan,  # MPa
            'v': np.nan,  # m³/kg
            's': np.nan,  # kJ/kg·K
            'h': np.nan,  # kJ/kg
            'u': np.nan,   # kJ/kg
            'x': 0 # Quality (dimensionless)
        }
    }

    return variables

def system_relations(vars):
    """
    Calculates all the system-level variables based on the known variables.
    
    Parameters:
    vars (dict): Main dictionaries containing the variables.
    
    Returns:
    dict: Updated main dictionary with calculated variables.
    """
    # ----------------------------------------------------
    # A) m_dot RELATIONS
    # ----------------------------------------------------
    if unknown(vars['m_dot']):
        # 1) If Qh_dot, qh known => m_dot = Qh_dot / qh
        if known(vars['Qh_dot']) and known(vars['qh']):
            vars['m_dot'] = vars['Qh_dot'] / vars['qh']
        
        # 2) If Qc_dot, qc known => m_dot = Qc_dot / qc
        elif known(vars['Qc_dot']) and known(vars['qc']):
            vars['m_dot'] = vars['Qc_dot'] / vars['qc']
        
        # 3) If Wc_dot, wc known => m_dot = Wc_dot / wc
        elif known(vars['Wc_dot']) and known(vars['wc']):
            vars['m_dot'] = vars['Wc_dot'] / vars['wc']

    # ----------------------------------------------------
    # B) Qh_dot, qh RELATIONS
    # ----------------------------------------------------
    if unknown(vars['qh']):
        # 1) If Qh_dot, m_dot known => qh = Qh_dot / m_dot
        if known(vars['Qh_dot']) and known(vars['m_dot']):
            vars['qh'] = vars['Qh_dot'] / vars['m_dot']
        
        # 2) If h3, h4 known => qh = h3 - h4
        elif known(vars['3']['h']) and known(vars['4']['h']):
            vars['qh'] = vars['3']['h'] - vars['4']['h']
    
    if unknown(vars['Qh_dot']):
        # 2) If Qc_dot, Wc_dot known => Qh_dot = Qc_dot + Wc_dot
        if known(vars['Qc_dot']) and known(vars['Wc_dot']):
            vars['Qh_dot'] = vars['Qc_dot'] + vars['Wc_dot']

        # 2) If qh, m_dot known => Qh_dot = qh * m_dot
        elif known(vars['qh']) and known(vars['m_dot']):
            vars['Qh_dot'] = vars['qh'] * vars['m_dot']
            
    # ----------------------------------------------------
    # C) Qc_dot, qc RELATIONS
    # ----------------------------------------------------
    if unknown(vars['qc']):
        # 1) If Qc_dot, m_dot known => qc = Qc_dot / m_dot
        if known(vars['Qc_dot']) and known(vars['m_dot']):
            vars['qc'] = vars['Qc_dot'] / vars['m_dot']
        
        # 2) If h2, h1 known => qc = h2 - h1
        elif known(vars['2']['h']) and known(vars['1']['h']):
            vars['qc'] = vars['2']['h'] - vars['1']['h']
    
    if unknown(vars['Qc_dot']):
        # 1) If Qh_dot, Wc_dot known => Qc_dot = Qh_dot - Wc_dot
        if known(vars['Qh_dot']) and known(vars['Wc_dot']):
            vars['Qc_dot'] = vars['Qh_dot'] - vars['Wc_dot']

        # 2) If qc, m_dot known => Qc_dot = qc * m_dot
        elif known(vars['qc']) and known(vars['m_dot']):  
            vars['Qc_dot'] = vars['qc'] * vars['m_dot']
    
    # ----------------------------------------------------
    # D) Wc_dot and wc RELATIONS
    # ----------------------------------------------------
    if unknown(vars['wc']):
        # 1) If Wc_dot, m_dot known => wc = Wc_dot / m_dot
        if known(vars['Wc_dot']) and known(vars['m_dot']):
            vars['wc'] = vars['Wc_dot'] / vars['m_dot']
        
        # 2) If h3, h2 known => wc = h3 - h2
        elif known(vars['3']['h']) and known(vars['2']['h']):
            vars['wc'] = vars['3']['h'] - vars['2']['h']

    if unknown(vars['Wc_dot']):
        # 1) If Qh_dot, Qc_dot known => Wc_dot = Qh_dot - Qc_dot
        #if known(vars['Qh_dot']) and known(vars['Qc_dot']):
        #    vars['Wc_dot'] = vars['Qh_dot'] - vars['Qc_dot']
        
        # 2) If wc, m_dot known => Wc_dot = wc*m_dot
        if known(vars['wc']) and known(vars['m_dot']):
            vars['Wc_dot'] = vars['wc'] * vars['m_dot']
    
    # ----------------------------------------------------
    # E) n (efficiency) and COP RELATIONS
    # ----------------------------------------------------    
    if unknown(vars['n']):
        # 1) If Wc_dot, Qh_dot known => n = Qh_dot / Wc_dot
        if known(vars['Qh_dot']) and known(vars['Wc_dot']):
            vars['n'] = vars['Qh_dot'] / vars['Wc_dot']

    if unknown(vars['COP_hp']):
        # 1) If Qh_dot, Wc_dot known => COP_hp = Qh_dot / Wc_dot
        if known(vars['Qh_dot']) and known(vars['Wc_dot']):
            vars['COP_hp'] = vars['Qh_dot'] / vars['Wc_dot']

    return vars

def equalize(var1, var2):
    """
    Sets two variables equal to each other if one is known and the other is unknown.
    
    Parameters:
    var1 (float): The first variable.
    var2 (float): The second variable.
    
    Returns:
    tuple: A tuple containing the updated variables.
    """
    if unknown(var1) and known(var2):
        var1 = var2
    elif unknown(var2) and known(var1):
        var2 = var1
    
    return var1, var2

def process_relations(vars):
    """
    Calculates the properties linked together by process relations.

    Parameters:
    vars (dict): Main dictionaries containing the variables.

    Returns:
    dict: Updated main dictionary with calculated variables.
    """
    # ----------------------------------------------------
    # A) PROCESS 1-2 Isobaric and isothermal
    # ----------------------------------------------------
    vars['1']['P'], vars['2']['P'] = equalize(vars['1']['P'], vars['2']['P'])
    vars['1']['T'], vars['2']['T'] = equalize(vars['1']['T'], vars['2']['T'])

    # -----------------------------------------------------
    # B) PROCESS 2-3 Isentropic
    # -----------------------------------------------------
    vars['2']['s'], vars['3']['s'] = equalize(vars['2']['s'], vars['3']['s'])
    
    # ------------------------------------------------------
    # C) PROCESS 3-3b Isobaric
    # ------------------------------------------------------
    vars['3']['P'], vars['3b']['P'] = equalize(vars['3']['P'], vars['3b']['P'])
    
    # ------------------------------------------------------
    # D) PROCESS 3b-4 Isobaric and isothermal
    # ------------------------------------------------------
    vars['3b']['P'], vars['4']['P'] = equalize(vars['3b']['P'], vars['4']['P'])
    vars['3b']['T'], vars['4']['T'] = equalize(vars['3b']['T'], vars['4']['T'])
    
    # ------------------------------------------------------
    # E) PROCESS 4-1 Isenthalpic
    # ------------------------------------------------------
    vars['4']['h'], vars['1']['h'] = equalize(vars['4']['h'], vars['1']['h'])
    
    return vars

def vars_from_x_and_PT(vars, known_var):
    x = vars['x']
    if known_var == 'P':
        table_base = 'Pressure'
        if unknown(vars['T']):
            vars['T'] = get_apdx_9ab(table_base, known_var, vars[known_var], 'T')
    
    elif known_var == 'T':
        table_base = 'Temperature'
        if unknown(vars['P']):
            vars['P'] = get_apdx_9ab(table_base, known_var, vars[known_var], 'P')

    for var in ['v', 'h', 's', 'u']:
        if unknown(vars[var]):
            var_f = get_apdx_9ab(table_base, known_var, vars[known_var], var + 'f')
            var_g = get_apdx_9ab(table_base, known_var, vars[known_var], var + 'g')
            vars[var] = (1 - x) * var_f + x * var_g
    
    return vars

def vars_from_x_and_quality_var(vars, known_var):
    """
    Fill in all state variables based on known quality (x) and one quality-dependent variable (h, s, or u).
    """
    assert known_var in ['v', 'h', 's', 'u'], "Known variable must be a quality-dependent property."

    x = vars['x']
    
    target_value = vars[known_var]
    table_base = 'Temperature'  # We'll search over T to match the known property
    
    if x == 0 or x == 1:
        # If x is 0 or 1, we can directly use the saturated properties
        T_sat = get_apdx_9ab(table_base, known_var + ('f' if x == 0 else 'g'), target_value, 'T')

    else:
        # If x is not 0 or 1, we need to find T such that the property matches
        def objective(T):
            var_f = get_apdx_9ab(table_base, 'T', T, known_var + 'f')
            var_g = get_apdx_9ab(table_base, 'T', T, known_var + 'g')
            interpolated = (1 - x) * var_f + x * var_g
            return interpolated - target_value

        # Root-finding bounds for T [°C] — Range of saturation temperature for R134a in the tables
        T_min, T_max = -24, 100

        if objective(T_min) * objective(T_max) >= 0:
            print(f'Known variable: {known_var} = {target_value}')
            print(f"f({T_min}) = {objective(T_min)}, f({T_max}) = {objective(T_max)}")
            raise ValueError("f(a) and f(b) must have different signs for root_scalar to work.")

        sol = root_scalar(objective, bracket=[T_min, T_max], method='brentq')
        T_sat = sol.root

    vars['T'] = T_sat

    # Now get the corresponding saturation pressure
    vars['P'] = get_apdx_9ab(table_base, 'T', T_sat, 'P')

    # Fill in other state variables using interpolation
    for var in ['v', 'h', 's', 'u']:
        if unknown(vars[var]):
            var_f = get_apdx_9ab(table_base, 'T', T_sat, var + 'f')
            var_g = get_apdx_9ab(table_base, 'T', T_sat, var + 'g')
            vars[var] = (1 - x) * var_f + x * var_g

    return vars
    
def x_from_PT_and_var(vars, known_var):
    """
    Compute the quality (x) based on one quality-dependent variable (h, s, u, or v)
    and either pressure (P) or temperature (T).
    """
    assert known_var in ['h', 's', 'u', 'v'], "Known variable must be a quality-dependent property."
    assert known(vars[known_var]), f"{known_var} must be known."
    assert known(vars['P']) or known(vars['T']), "Either P or T must be known."

    # Decide the table base and lookup key
    print(f'T = {vars["T"]}, P = {vars["P"]}, {known_var} = {vars[known_var]}')
    if known(vars['T']):
        table_base = 'Temperature'
        lookup_value = vars['T']
    else:
        table_base = 'Pressure'
        lookup_value = vars['P']

    var_f = get_apdx_9ab(table_base, table_base[0], lookup_value, known_var + 'f')
    var_g = get_apdx_9ab(table_base, table_base[0], lookup_value, known_var + 'g')

    x = (vars[known_var] - var_f) / (var_g - var_f)
    vars['x'] = x

    return vars

def saturated_state(variables, state):
    """
    Calculates the properties of a saturated state based on the given variables.
    
    Parameters:
    variables (dict): Main dictionaries containing the variables.
    state (int): The state number (1, 2, 3, or 4).
    
    Returns:
    dict: Updated main dictionary with calculated variables.
    """
    print(f'Calculating state {state}')
    vars = variables[state]
    for var in vars:
        if known(vars[var]) and known(vars['x']) and (var == 'P' or var == 'T'):
            # Get the saturation properties based on the known variable
            vars = vars_from_x_and_PT(vars, var)
        elif known(vars[var]) and (var == 'v' or var == 'h' or var == 's' or var == 'u'):
            # Get the saturation properties based on the known variable
            if known(vars['x']):
                vars = vars_from_x_and_quality_var(vars, var)
            elif known(vars['P']) or known(vars['T']):
                vars = x_from_PT_and_var(vars, var)
    variables[state] = vars

    return variables

def superheated_state(variables, state):
    """
    Calculates the properties of a superheated state based on the given variables.

    Parameters:
    variables (dict): Main dictionaries containing the variables.
    state (str): The index of the state to update (1, 2, 3, 4, or 3b).

    Returns:
    dict: Updated main dictionary with calculated variables.
    """
    vars = variables[state]

    known_vars = [var for var in ['P', 'T', 'v', 'h', 's', 'u'] if known(vars[var])]
    if len(known_vars) < 2:
        return variables  # Not enough information yet

    # Priority: use (P, T) if both are known
    if known(vars['P']) and known(vars['T']):
        input_pair = ('P', 'T')
        input_values = (vars['P'], vars['T'])

    # Try combinations with quality-dependent variables
    else:
        # Pick the first pair we can find
        pairs = [('P', 'h'), ('P', 's'), ('P', 'u'), ('P', 'v'),
                 ('T', 'h'), ('T', 's'), ('T', 'u'), ('T', 'v')]
        for var1, var2 in pairs:
            if known(vars[var1]) and known(vars[var2]):
                input_pair = (var1, var2)
                input_values = (vars[var1], vars[var2])
                break
        else:
            return variables  # No valid pair found yet

    # Compute all remaining variables
    for var in ['P', 'T', 'v', 'u', 'h', 's']:
        if unknown(vars[var]):
            vars[var] = get_apdx_9c(input_pair, input_values, var)

    variables[state] = vars
    return variables


def solve_r_rankine_cycle(variables):
    """
    Solves the reverse rankine cycle using the known variables and calculates the unknown variables.
    
    Parameters:
    variables (dict): Main dictionaries containing the variables.
    
    Returns:
    dict: Updated main dictionary with calculated variables.
    """
    
    def count_nans(variables):
        return sum(
            unknown(item) 
            for key, val in variables.items()       # top level
            for item in (val.values() if isinstance(val, dict) else [val])
        )
    
    previous_nan_count = count_nans(variables)
    counter = 0
    while True:
        # Calculate system-level variables such as m_dot, Qh_dot, Qc_dot, Wc_dot, etc.
        variables = system_relations(variables)

        # Calculates the properties linked together by process relations
        variables = process_relations(variables)
        
        # Calculate the properties of the saturated states
        for state in ['1', '2', '3b', '4']:
            variables = saturated_state(variables, state)
        
        # Calculate the properties of the superheated state
        variables = superheated_state(variables, '3')

        current_nan_count = count_nans(variables)
        counter += 1
        print(counter, current_nan_count)
        if current_nan_count == 0 or \
        current_nan_count >= previous_nan_count:
            break
        previous_nan_count = current_nan_count
    
    return variables


vars = define_empty_variables()

# give the solver *one* piece of information per state
vars['2']['T'] = -8  # K
vars['3']['P'] = 0.8 # MPa
vars['4']['h'] = 200 # kJ/kg
vars['m_dot'] = 0.1 # kg/s

solved = solve_r_rankine_cycle(vars)
print(solved)
print(solved['Wc_dot'], solved['Qh_dot'])