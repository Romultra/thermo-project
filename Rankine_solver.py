import numpy as np
from apdx_functions import *

# Diesel cycle solver
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
    list: A list of dictionaries containing the initialized variables.
    """
    # Initialize all variables to NaN
    variables = {
        'R':  np.nan,  # Specific gas constant (kJ/(kg·K)
        'gamma':np.nan,# Heat capacity ratio (dimensionless)
        'm_dot':np.nan, # Mass flow rate (kg/s)
        'Qh_dot': np.nan,  # Heat addition rate (kW or kJ/s)
        'qh': np.nan,  # Heat addition (kJ/kg)
        'Qc_dot': np.nan,  # Heat rejection rate (kW or kJ/s)
        'qc': np.nan,  # Heat rejection (kJ/kg)
        'Wc_dot': np.nan,  # Work done by the compressor (kW or kJ/s)
        'Wc': np.nan,  # Work done by the compressor (kJ/kg)
        'n':  np.nan,  # Efficiency (dimensionless)
        'COP_hp': np.nan, # Coefficient of performance (dimensionless)

        'T1': np.nan,  # K
        'P1': np.nan,  # kPa
        'v1': np.nan,  # m³/kg
        's1': np.nan,  # kJ/kg·K
        'h1': np.nan,  # kJ/kg
        'u1': np.nan,  # kJ/kg
        'x1': np.nan,  # Quality (dimensionless)

        'T2': np.nan,  # K
        'P2': np.nan,  # kPa
        'v2': np.nan,  # m³/kg
        's2': np.nan,  # kJ/kg·K
        'h2': np.nan,  # kJ/kg
        'u2': np.nan,  # kJ/kg

        'T3': np.nan,  # K
        'P3': np.nan,  # kPa
        'v3': np.nan,  # m³/kg
        's3': np.nan,  # kJ/kg·K
        'h3': np.nan,  # kJ/kg
        'u3': np.nan,  # kJ/kg

        'T3b': np.nan,  # K
        'P3b': np.nan,  # kPa
        'v3b': np.nan,  # m³/kg
        's3b': np.nan,  # kJ/kg·K
        'h3b': np.nan,  # kJ/kg
        'u3b': np.nan,  # kJ/kg

        'T4': np.nan,  # K
        'P4': np.nan,  # kPa
        'v4': np.nan,  # m³/kg
        's4': np.nan,  # kJ/kg·K
        'h4': np.nan,  # kJ/kg
        'u4': np.nan   # kJ/kg
    }

    return variables

def system_variables(vars):
    """
    Calculates all the system-level variables based on the known variables.
    
    Parameters:
    vars (list): List of dictionaries containing the variables.
    
    Returns:
    list: Updated list of dictionaries with calculated variables.
    """
    # ----------------------------------------------------
    # A) m_dot RELATIONS
    # ----------------------------------------------------
    if unknown(vars['m_dot']):
        # 1) If Qh_dot, qh known => m_dot = Qh_dot / qh
        if known(vars['Qh_dot']) and known(vars['qh']):
            vars['m_dot'] = vars['Qh_dot'] / vars['qh']
        
        # 2) If Qc_dot, qc known => m_dot = Qc_dot / qc
        if known(vars['Qc_dot']) and known(vars['qc']):
            vars['m_dot'] = vars['Qc_dot'] / vars['qc']
        
        # 3) If Wc_dot, Wc known => m_dot = Wc_dot / Wc
        if known(vars['Wc_dot']) and known(vars['Wc']):
            vars['m_dot'] = vars['Wc_dot'] / vars['Wc']

    # ----------------------------------------------------
    # B) Qh_dot, qh RELATIONS
    # ----------------------------------------------------
    if unknown(vars['qh']):
        # 1) If Qh_dot, m_dot known => qh = Qh_dot / m_dot
        if known(vars['Qh_dot']) and known(vars['m_dot']):
            vars['qh'] = vars['Qh_dot'] / vars['m_dot']
        
        # 2) If h3, h4 known => qh = h3 - h4
        if known(vars['h3']) and known(vars['h4']):
            vars['qh'] = vars['h3'] - vars['h4']
    
    if unknown(vars['Qh_dot']):
        # 2) If Qc_dot, Wc_dot known => Qh_dot = Qc_dot + Wc_dot
        if known(vars['Qc_dot']) and known(vars['Wc_dot']):
            vars['Qh_dot'] = vars['Qc_dot'] + vars['Wc_dot']

        # 2) If qh, m_dot known => Qh_dot = qh * m_dot
        if known(vars['qh']) and known(vars['m_dot']):
            vars['Qh_dot'] = vars['qh'] * vars['m_dot']
            
    # ----------------------------------------------------
    # C) Qc_dot, qc RELATIONS
    # ----------------------------------------------------
    if unknown(vars['qc']):
        # 1) If Qc_dot, m_dot known => qc = Qc_dot / m_dot
        if known(vars['Qc_dot']) and known(vars['m_dot']):
            vars['qc'] = vars['Qc_dot'] / vars['m_dot']
        
        # 2) If h2, h1 known => qc = h2 - h1
        if known(vars['h2']) and known(vars['h1']):
            vars['qc'] = vars['h2'] - vars['h1']
    
    if unknown(vars['Qc_dot']):
        # 1) If Qh_dot, Wc_dot known => Qc_dot = Qh_dot - Wc_dot
        if known(vars['Qh_dot']) and known(vars['Wc_dot']):
            vars['Qc_dot'] = vars['Qh_dot'] - vars['Wc_dot']

        # 2) If qc, m_dot known => Qc_dot = qc * m_dot
        if known(vars['qc']) and known(vars['m_dot']):  
            vars['Qc_dot'] = vars['qc'] * vars['m_dot']
    
    # ----------------------------------------------------
    # D) Wc_dot and wc RELATIONS
    # ----------------------------------------------------
    if unknown(vars['wc']):
        # 1) If Wc_dot, m_dot known => wc = Wc_dot / m_dot
        if known(vars['Wc_dot']) and known(vars['m_dot']):
            vars['wc'] = vars['Wc_dot'] / vars['m_dot']
        
        # 2) If h3, h2 known => wc = h3 - h2
        if known(vars['h3']) and known(vars['h2']):
            vars['wc'] = vars['h3'] - vars['h2']

    if unknown(vars['Wc_dot']):
        # 1) If Qh_dot, Qc_dot known => Wc_dot = Qh_dot - Qc_dot
        if known(vars['Qh_dot']) and known(vars['Qc_dot']):
            vars['Wc_dot'] = vars['Qh_dot'] - vars['Qc_dot']
        
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

def solve_r_rankine_cycle(variables):
    """
    Solves the reverse rankine cycle using the known variables and calculates the unknown variables.
    
    Parameters:
    variables (list): List of dictionaries containing the variables.
    
    Returns:
    list: Updated list of dictionaries with calculated variables.
    """
    
    def count_nans(variables):
        return sum(unknown(value) for state in variables for value in state.values())
    
    previous_nan_count = count_nans(variables)
    counter = 0
    while True:
        # Calculate system-level variables such as m_dot, Qh_dot, Qc_dot, Wc_dot, etc.
        variables = system_variables(variables)
        
        # Process 1-2:
        variables = 'Placeholder for process 1-2 calculation'
        
        # Process 2-3:
        variables = 'Placeholder for process 2-3 calculation'

        # Process 3-3b:
        variables = 'Placeholder for process 3-3b calculation'

        # Process 3b-4:
        variables = 'Placeholder for process 3b-4 calculation'

        # Process 4-1:
        variables = 'Placeholder for process 4-1 calculation'
        
        current_nan_count = count_nans(variables)
        counter += 1
        print(counter, current_nan_count)
        if current_nan_count == previous_nan_count:
            break
        previous_nan_count = current_nan_count
    
    return variables