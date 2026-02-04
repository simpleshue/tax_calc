import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# =============================================================================
# GLOBAL CONFIGURATION (2025 TAX YEAR - MARRIED FILING JOINTLY)
# =============================================================================
TAX_CONFIG = {
    "STANDARD_DEDUCTION": 30000, # Increased for 2025
    "NIIT_THRESHOLD": 250000,    # Not indexed for inflation
    "NIIT_RATE": 0.038,
    "FEIE_LIMIT": 130000,        # 2025 Limit
    
    # 2025 Ordinary Income Brackets (MFJ)
    "ORDINARY_BRACKETS": [
        (23850, 0.10),
        (96950, 0.12),
        (206700, 0.22),
        (394600, 0.24),
        (501050, 0.32),
        (751600, 0.35),
        (float('inf'), 0.37)
    ],
    
    # 2025 Long Term Capital Gains Brackets (MFJ)
    "LTCG_BRACKETS": [
        (96700, 0.00),
        (600050, 0.15),
        (float('inf'), 0.20)
    ],
    
    "PENALTY_FREE_AGE": 60, 
    "RMD_START_AGE": 73,
    "RMD_DISTRIBUTION_PERIODS": {
        73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2,
        81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7,
        89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9, 100: 6.4
    },
    
    # Social Security Taxation Thresholds (Not Indexed)
    "SS_BASE_LOWER": 32000,
    "SS_BASE_UPPER": 44000,
    
    # 2025 Medicare IRMAA Brackets (Married Filing Jointly)
    # Based on 2023 MAGI. Amounts are Monthly Surcharges per person.
    # (Threshold, Part B Surcharge, Part D Surcharge)
    "IRMAA_BRACKETS": [
        (212000, 0.0, 0.0),             # Base
        (266000, 74.00, 13.70),         # Tier 1
        (332000, 185.00, 35.30),        # Tier 2
        (398000, 296.00, 57.00),        # Tier 3
        (750000, 407.00, 78.60),        # Tier 4
        (float('inf'), 444.00, 85.80)   # Tier 5
    ],
    
    # NJ State Tax Brackets (2024/2025 MFJ - Generally Static)
    "NJ_BRACKETS": [
        (20000, 0.014),
        (50000, 0.0175),
        (70000, 0.0245),
        (80000, 0.035),
        (150000, 0.05525),
        (500000, 0.0637),
        (1000000, 0.0897),
        (float('inf'), 0.1075)
    ],
    "NJ_RETIREMENT_AGE": 62,
    "NJ_EXCLUSION_LIMIT_FULL": 100000, 
    "NJ_EXCLUSION_LIMIT_PARTIAL_1": 0.50, 
    "NJ_EXCLUSION_LIMIT_PARTIAL_2": 0.25, 
    "NJ_INCOME_CLIFF": 150000 
}

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExpenseParams:
    """Configuration for living expenses for a specific period."""
    method: str = 'fixed_inflation'  
    base_annual_expense: float = 0.0 
    expense_ratio: float = 0.0       
    max_annual_expense: float = float('inf') 

@dataclass
class SimParameters:
    """Inputs for the retirement simulation."""
    # Mandatory fields
    current_age: int
    retirement_age: int
    death_age: int
    cash_savings: float
    taxable_account_basis: float 
    pretax_balance: float        
    roth_balance: float          
    
    # Growth & Economics
    retirement_return_rate: float  
    taxable_return_rate: float     
    inflation_rate: float        

    # Income & Contributions
    annual_salary: float
    annual_pretax_contribution: float  
    annual_roth_contribution: float    
    
    # Income (Post-retirement)
    social_security: float       
    social_security_start_age: int
    
    # Expenses
    default_expense_params: ExpenseParams
    
    # Pretax Strategic Withdrawal Ratio
    annual_pretax_withdrawal_ratio: float 

    # Roth Conversion Strategy
    do_roth_conversions: bool
    conversion_start_age: int
    conversion_end_age: int
    annual_conversion_amount: float

    # Optional fields
    start_year: int = field(default_factory=lambda: datetime.now().year)
    return_overrides: Dict[int, Dict[str, float]] = field(default_factory=dict)
    expense_overrides: List[Tuple[int, int, ExpenseParams]] = field(default_factory=list)
    salary_overrides: List[Tuple[int, int, float]] = field(default_factory=list)
    
    # Abroad Years: List of (start_year, end_year)
    abroad_years: List[Tuple[int, int]] = field(default_factory=list)

    # Housing Costs
    mortgage_monthly_payment: float = 0.0
    mortgage_remaining_years: int = 0
    property_tax_annual: float = 0.0
    
    # Healthcare Costs by Age Range: List of (start_age, end_age, annual_cost_today_dollars)
    # Example: [(55, 64, 24000), (65, 85, 12000), (86, 90, 100000)]
    healthcare_costs: List[Tuple[int, int, float]] = field(default_factory=list)

@dataclass
class YearData:
    """Snapshot of financial state for a single year."""
    age: int
    calendar_year: int
    is_abroad: bool 
    
    start_pretax: float
    start_roth: float
    start_cash: float
    start_taxable_basis: float
    start_total: float
    
    salary_income: float
    social_security_income: float
    social_security_taxable: float 
    growth_pretax: float
    growth_roth: float
    growth_cash: float
    
    contribution_pretax: float
    contribution_roth: float
    rmd_amount: float              
    strategic_pretax_withdrawal: float 
    roth_conversion_amount: float
    
    available_cash_flow_pre_tax_and_expense: float      
    surplus_saved: float           
    portfolio_withdrawal_needed: float 
    
    expenses: float
    irmaa_cost: float 
    mortgage_payment: float 
    property_tax_payment: float 
    healthcare_cost: float # New: Explicit Healthcare (Base Premiums/LTC)
    expenses_total: float 
    
    withdrawal_pretax_extra: float 
    withdrawal_roth: float
    withdrawal_cash: float
    capital_gains_realized: float
    
    # Tax Result (Breakdown)
    tax_ordinary: float            
    tax_ltcg: float                
    tax_niit: float  
    tax_state: float             
    taxes_total: float             
    effective_tax_rate: float
    
    # Income Tracking for IRMAA Lookback
    magi: float 
    prev_year_magi: float 
    
    # Closing State
    end_pretax: float
    end_roth: float
    end_cash: float
    end_taxable_basis: float
    end_total: float
    
    cumulative_inflation: float = 1.0

# =============================================================================
# CALCULATION ENGINE
# =============================================================================

class TaxCalculator:
    """Logic to compute IRS and State tax liability."""
    
    @staticmethod
    def calculate_rmd(age: int, pretax_balance: float) -> float:
        if age < TAX_CONFIG["RMD_START_AGE"]:
            return 0.0
        period = TAX_CONFIG["RMD_DISTRIBUTION_PERIODS"].get(age, 6.0)
        return pretax_balance / period

    @staticmethod
    def calculate_taxable_ss(ss_benefit: float, other_income: float) -> float:
        if ss_benefit <= 0: return 0.0
        provisional_income = other_income + (0.5 * ss_benefit)
        if provisional_income <= TAX_CONFIG["SS_BASE_LOWER"]:
            taxable_ss = 0.0
        elif provisional_income <= TAX_CONFIG["SS_BASE_UPPER"]:
            taxable_ss = min(0.5 * ss_benefit, 0.5 * (provisional_income - TAX_CONFIG["SS_BASE_LOWER"]))
        else:
            temp1 = 0.85 * (provisional_income - TAX_CONFIG["SS_BASE_UPPER"])
            temp2 = min(0.5 * ss_benefit, 6000) 
            taxable_ss = min(0.85 * ss_benefit, temp1 + temp2)
        return taxable_ss

    @staticmethod
    def calculate_irmaa_surcharge(magi_2_years_ago: float) -> float:
        monthly_b = 0.0
        monthly_d = 0.0
        for threshold, sur_b, sur_d in TAX_CONFIG["IRMAA_BRACKETS"]:
            if magi_2_years_ago <= threshold:
                monthly_b = sur_b
                monthly_d = sur_d
                break
        return (monthly_b + monthly_d) * 12 * 2

    @staticmethod
    def calculate_ordinary_tax(income: float) -> float:
        tax = 0.0
        taxable_ord = max(0, income - TAX_CONFIG["STANDARD_DEDUCTION"])
        prev_limit = 0
        for limit, rate in TAX_CONFIG["ORDINARY_BRACKETS"]:
            if taxable_ord > prev_limit:
                chunk = min(taxable_ord, limit) - prev_limit
                tax += chunk * rate
                prev_limit = limit
            else:
                break
        return tax

    @staticmethod
    def calculate_ltcg_tax(taxable_ordinary_income: float, gains: float) -> float:
        tax = 0.0
        remaining_gains = gains
        current_floor = taxable_ordinary_income
        for limit, rate in TAX_CONFIG["LTCG_BRACKETS"]:
            if remaining_gains <= 0: break
            if current_floor < limit:
                eligible_chunk = min(remaining_gains, limit - current_floor)
                tax += eligible_chunk * rate
                remaining_gains -= eligible_chunk
                current_floor += eligible_chunk
        return tax

    @staticmethod
    def calculate_nj_tax(salary: float, pension_income: float, capital_gains: float, age: int) -> float:
        gross_income = salary + pension_income + capital_gains
        exclusion = 0.0
        if age >= TAX_CONFIG["NJ_RETIREMENT_AGE"]:
            if gross_income <= 100000:
                exclusion = min(pension_income, TAX_CONFIG["NJ_EXCLUSION_LIMIT_FULL"])
            elif gross_income <= 125000:
                exclusion = min(pension_income, TAX_CONFIG["NJ_EXCLUSION_LIMIT_FULL"] * TAX_CONFIG["NJ_EXCLUSION_LIMIT_PARTIAL_1"])
            elif gross_income <= 150000:
                exclusion = min(pension_income, TAX_CONFIG["NJ_EXCLUSION_LIMIT_FULL"] * TAX_CONFIG["NJ_EXCLUSION_LIMIT_PARTIAL_2"])
            else:
                exclusion = 0.0
        
        nj_taxable_income = max(0.0, gross_income - exclusion)
        tax = 0.0
        prev_limit = 0
        for limit, rate in TAX_CONFIG["NJ_BRACKETS"]:
            if nj_taxable_income > prev_limit:
                chunk = min(nj_taxable_income, limit) - prev_limit
                tax += chunk * rate
                prev_limit = limit
            else:
                break
        return tax

    @classmethod
    def calculate_detailed_tax(cls, ordinary_income: float, long_term_gains: float, 
                               nj_salary: float, nj_pension: float, nj_gains: float, age: int) -> Dict[str, float]:
        # Federal
        ord_tax = cls.calculate_ordinary_tax(ordinary_income)
        taxable_ord_base = max(0, ordinary_income - TAX_CONFIG["STANDARD_DEDUCTION"])
        ltcg_tax = cls.calculate_ltcg_tax(taxable_ord_base, long_term_gains)
        magi = ordinary_income + long_term_gains
        niit_tax = 0.0
        if magi > TAX_CONFIG["NIIT_THRESHOLD"]:
            niit_subject = min(long_term_gains, magi - TAX_CONFIG["NIIT_THRESHOLD"])
            niit_tax = niit_subject * TAX_CONFIG["NIIT_RATE"]
            
        # State (NJ)
        state_tax = cls.calculate_nj_tax(nj_salary, nj_pension, nj_gains, age)
            
        return {
            "ordinary": ord_tax,
            "ltcg": ltcg_tax,
            "niit": niit_tax,
            "state": state_tax,
            "total": ord_tax + ltcg_tax + niit_tax + state_tax
        }

def calculate_year(prev_state: YearData, params: SimParameters) -> YearData:
    curr_age = prev_state.age + 1
    curr_year = prev_state.calendar_year + 1
    
    # 0. Check Status (Abroad?)
    is_abroad = False
    for start, end in params.abroad_years:
        if start <= curr_year <= end:
            is_abroad = True
            break
    
    # 1. Returns & Inflation
    ret_rate = params.retirement_return_rate
    tax_rate = params.taxable_return_rate
    infl_rate = params.inflation_rate
    if curr_year in params.return_overrides:
        overrides = params.return_overrides[curr_year]
        ret_rate = overrides.get('retirement_return_rate', ret_rate)
        tax_rate = overrides.get('taxable_return_rate', tax_rate)
        infl_rate = overrides.get('inflation_rate', infl_rate)

    new_cumulative_inflation = prev_state.cumulative_inflation * (1 + infl_rate)
    inf_factor = new_cumulative_inflation

    # 2. Growth
    pretax = prev_state.end_pretax * (1 + ret_rate)
    roth = prev_state.end_roth * (1 + ret_rate)
    cash = prev_state.end_cash * (1 + tax_rate)
    growth_p = prev_state.end_pretax * ret_rate
    growth_r = prev_state.end_roth * ret_rate
    growth_c = prev_state.end_cash * tax_rate
    taxable_basis = prev_state.end_taxable_basis
    
    # 3. Income
    base_salary = None
    for start_yr, end_yr, sal in params.salary_overrides:
        if start_yr <= curr_year <= end_yr:
            base_salary = sal
            break
    if base_salary is None:
        if curr_age >= params.retirement_age:
            base_salary = 0.0
        else:
            base_salary = params.annual_salary
            
    salary = base_salary * inf_factor
    is_working = (base_salary > 0)
    
    ss = params.social_security * inf_factor if curr_age >= params.social_security_start_age else 0.0
    
    target_pretax_contrib = params.annual_pretax_contribution * inf_factor
    contrib_pretax = min(target_pretax_contrib, salary) if is_working else 0.0
    
    target_roth_contrib = params.annual_roth_contribution * inf_factor
    contrib_roth = min(target_roth_contrib, max(0.0, salary - contrib_pretax)) if is_working else 0.0
    
    # --- Federal Taxable Salary Logic with FEIE ---
    taxable_salary_base = max(0, salary - contrib_pretax)
    
    feie_deduction = 0.0
    if is_abroad and is_working:
        feie_limit_adj = TAX_CONFIG["FEIE_LIMIT"] * inf_factor
        feie_deduction = min(taxable_salary_base, feie_limit_adj)
    
    federal_taxable_salary = max(0, taxable_salary_base - feie_deduction)
    
    pretax += contrib_pretax
    roth += contrib_roth

    # 4. Forced Pretax
    portfolio_total = pretax + roth + cash
    rmd_amt = TaxCalculator.calculate_rmd(curr_age, pretax)
    strategic_ratio_amt = portfolio_total * params.annual_pretax_withdrawal_ratio if (curr_age >= TAX_CONFIG["PENALTY_FREE_AGE"] and pretax > 0) else 0.0
    total_forced_pretax_withdrawal = max(rmd_amt, strategic_ratio_amt)
    actual_forced_pretax_withdrawal = min(total_forced_pretax_withdrawal, pretax)
    pretax = max(0.0, pretax - actual_forced_pretax_withdrawal)

    # 5. Expenses & Liabilities
    # Housing
    mortgage_payment = 0.0
    elapsed_years = curr_year - params.start_year
    if elapsed_years < params.mortgage_remaining_years:
        mortgage_payment = params.mortgage_monthly_payment * 12
    
    property_tax_payment = params.property_tax_annual * inf_factor
    
    # Healthcare Costs (Base + LTC) - Matches specific age ranges
    # Note: IRMAA is calculated separately and stacked on top.
    healthcare_base_cost = 0.0
    for h_start, h_end, h_cost in params.healthcare_costs:
        if h_start <= curr_age <= h_end:
            healthcare_base_cost = h_cost * inf_factor
            break
            
    # IRMAA (Surcharge)
    irmaa_cost = 0.0
    if curr_age >= 65:
        magi_lagged = prev_state.prev_year_magi
        irmaa_cost = TaxCalculator.calculate_irmaa_surcharge(magi_lagged)

    # Lifestyle
    active_exp_params = params.default_expense_params
    for start_yr, end_yr, exp_override in params.expense_overrides:
        if start_yr <= curr_year <= end_yr:
            active_exp_params = exp_override
            break

    if active_exp_params.method == 'asset_ratio':
        start_total_for_ratio = pretax + roth + cash
        base_assets = start_total_for_ratio + salary + ss + actual_forced_pretax_withdrawal
        target_exp = base_assets * active_exp_params.expense_ratio
        max_exp_allowed = active_exp_params.max_annual_expense * inf_factor
        target_exp = min(target_exp, max_exp_allowed)
    else:
        target_exp = active_exp_params.base_annual_expense * inf_factor
    
    # Total Cash Outflow Target
    total_target_outflow = target_exp + healthcare_base_cost + irmaa_cost + mortgage_payment + property_tax_payment

    # 7. Conversions
    conv_amt = 0.0
    if (params.do_roth_conversions and 
        params.conversion_start_age <= curr_age <= params.conversion_end_age and pretax > 0):
        conv_amt = min(params.annual_conversion_amount, pretax)
        pretax -= conv_amt
        roth += conv_amt

    # 8. Tax Iteration
    current_tax_results = {"ordinary": 0, "ltcg": 0, "niit": 0, "state": 0, "total": 0}
    w_cash, w_roth, w_pretax_extra, realized_gains = 0.0, 0.0, 0.0, 0.0
    current_taxable_ss = 0.0
    amount_spent_from_forced = 0.0 
    calculated_magi = 0.0 
    
    base_cash_flow_gross = taxable_salary_base + ss - contrib_roth

    for _ in range(5):
        total_needed = total_target_outflow + current_tax_results["total"]
        remaining_needed = max(0.0, total_needed - base_cash_flow_gross)
        
        _w_cash, _w_roth, _w_pretax_extra, _gains = 0.0, 0.0, 0.0, 0.0
        
        # Priority 1: Cash
        if remaining_needed > 0 and cash > 0:
            _w_cash = min(remaining_needed, cash)
            gain_ratio = max(0.0, (cash - taxable_basis) / cash) if cash > 0 else 0
            _gains = _w_cash * gain_ratio
            remaining_needed -= _w_cash

        # Priority 2: Forced Pretax
        _spent_forced = min(remaining_needed, actual_forced_pretax_withdrawal)
        remaining_needed -= _spent_forced
        
        # Priority 3: Roth
        if remaining_needed > 0 and roth > 0:
            _w_roth = min(remaining_needed, roth)
            remaining_needed -= _w_roth
            
        # Priority 4: Extra Pretax
        if remaining_needed > 0 and pretax > 0:
            _w_pretax_extra = min(remaining_needed, pretax)
            remaining_needed -= _w_pretax_extra
            
        # Federal Calcs
        fed_pension_income = actual_forced_pretax_withdrawal + _w_pretax_extra + conv_amt
        other_income_fed = federal_taxable_salary + fed_pension_income
        current_taxable_ss = TaxCalculator.calculate_taxable_ss(ss, other_income_fed)
        ord_inc_fed = other_income_fed + current_taxable_ss
        
        # State Calcs (NJ)
        if is_abroad:
            nj_salary_in = 0.0
            nj_pension_in = 0.0
            nj_gains_in = 0.0
        else:
            nj_salary_in = taxable_salary_base
            nj_pension_in = fed_pension_income
            nj_gains_in = _gains
        
        new_tax_dict = TaxCalculator.calculate_detailed_tax(
            ordinary_income=ord_inc_fed, 
            long_term_gains=_gains,
            nj_salary=nj_salary_in,
            nj_pension=nj_pension_in,
            nj_gains=nj_gains_in,
            age=curr_age
        )
        
        calculated_magi = ord_inc_fed + _gains
        
        if abs(new_tax_dict["total"] - current_tax_results["total"]) < 1.0:
            current_tax_results = new_tax_dict
            w_cash, w_roth, w_pretax_extra, realized_gains = _w_cash, _w_roth, _w_pretax_extra, _gains
            amount_spent_from_forced = _spent_forced
            break
        current_tax_results = new_tax_dict
        w_cash, w_roth, w_pretax_extra, realized_gains = _w_cash, _w_roth, _w_pretax_extra, _gains
        amount_spent_from_forced = _spent_forced

    # 9. Finalize
    net_after_tax_needs = total_target_outflow + current_tax_results["total"]
    surplus_to_cash = 0.0
    
    forced_remainder_to_roth = actual_forced_pretax_withdrawal - amount_spent_from_forced
    roth += forced_remainder_to_roth
    
    if base_cash_flow_gross > net_after_tax_needs:
        surplus_to_cash = base_cash_flow_gross - net_after_tax_needs
        cash += surplus_to_cash
        taxable_basis += surplus_to_cash
        w_cash, w_roth, w_pretax_extra = 0, 0, 0
    else:
        if w_cash > 0:
            basis_ratio = taxable_basis / (cash + w_cash) if (cash + w_cash) > 0 else 0
            taxable_basis -= (w_cash * basis_ratio)
        cash -= w_cash
        roth -= w_roth
        pretax -= w_pretax_extra

    total_gross_taxable = (federal_taxable_salary + current_taxable_ss + conv_amt + actual_forced_pretax_withdrawal + w_pretax_extra + realized_gains)
    eff_rate = (current_tax_results["total"] / total_gross_taxable) if total_gross_taxable > 0 else 0

    return YearData(
        age=curr_age, calendar_year=curr_year, is_abroad=is_abroad,
        start_pretax=prev_state.end_pretax, start_roth=prev_state.end_roth, start_cash=prev_state.end_cash,
        start_taxable_basis=prev_state.end_taxable_basis, start_total=prev_state.end_total,
        salary_income=salary, social_security_income=ss, social_security_taxable=current_taxable_ss,
        growth_pretax=growth_p, growth_roth=growth_r, growth_cash=growth_c,
        contribution_pretax=contrib_pretax, contribution_roth=contrib_roth,
        rmd_amount=rmd_amt, strategic_pretax_withdrawal=actual_forced_pretax_withdrawal,
        available_cash_flow_pre_tax_and_expense=base_cash_flow_gross + actual_forced_pretax_withdrawal,
        surplus_saved=surplus_to_cash, 
        portfolio_withdrawal_needed=max(0.0, net_after_tax_needs - (base_cash_flow_gross + actual_forced_pretax_withdrawal)),
        expenses=target_exp, irmaa_cost=irmaa_cost,
        mortgage_payment=mortgage_payment, property_tax_payment=property_tax_payment,
        healthcare_cost=healthcare_base_cost, # New
        expenses_total=total_target_outflow,
        roth_conversion_amount=conv_amt + forced_remainder_to_roth,
        withdrawal_pretax_extra=w_pretax_extra, withdrawal_roth=w_roth, withdrawal_cash=w_cash,
        capital_gains_realized=realized_gains, 
        tax_ordinary=current_tax_results["ordinary"],
        tax_ltcg=current_tax_results["ltcg"], 
        tax_niit=current_tax_results["niit"],
        tax_state=current_tax_results["state"],
        taxes_total=current_tax_results["total"], 
        effective_tax_rate=eff_rate,
        end_pretax=pretax, end_roth=roth, end_cash=cash, end_taxable_basis=taxable_basis, end_total=pretax + roth + cash,
        cumulative_inflation=new_cumulative_inflation,
        magi=calculated_magi, prev_year_magi=prev_state.magi
    )

def run_simulation(params: SimParameters) -> pd.DataFrame:
    init = YearData(
        age=params.current_age, calendar_year=params.start_year, is_abroad=False,
        start_pretax=0, start_roth=0, start_cash=0, start_taxable_basis=0, start_total=0,
        salary_income=0, social_security_income=0, social_security_taxable=0,
        growth_pretax=0, growth_roth=0, growth_cash=0,
        contribution_pretax=0, contribution_roth=0, rmd_amount=0, strategic_pretax_withdrawal=0,
        available_cash_flow_pre_tax_and_expense=0, surplus_saved=0, portfolio_withdrawal_needed=0,
        expenses=0, irmaa_cost=0, mortgage_payment=0, property_tax_payment=0, healthcare_cost=0, expenses_total=0, 
        roth_conversion_amount=0, withdrawal_pretax_extra=0, withdrawal_roth=0, withdrawal_cash=0,
        capital_gains_realized=0, 
        tax_ordinary=0, tax_ltcg=0, tax_niit=0, tax_state=0, taxes_total=0, effective_tax_rate=0,
        end_pretax=params.pretax_balance, end_roth=params.roth_balance, end_cash=params.cash_savings,
        end_taxable_basis=params.taxable_account_basis,
        end_total=params.pretax_balance + params.roth_balance + params.cash_savings,
        cumulative_inflation=1.0, magi=0.0, prev_year_magi=0.0
    )
    history = [init]
    curr = init
    while curr.age < params.death_age and curr.end_total > 0:
        curr = calculate_year(curr, params)
        history.append(curr)
    return pd.DataFrame([vars(d) for d in history])
