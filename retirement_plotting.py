import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import retirement_simulation


def add_milestones_to_fig(fig, df_results, params, x_axis='calendar_year'):
    """Helper to add vertical milestone lines and abroad regions to a plotly figure."""
    retirement_yr = params.start_year + (params.retirement_age - params.current_age)
    ss_yr = params.start_year + (params.social_security_start_age - params.current_age)
    penalty_free_yr = params.start_year + (60 - params.current_age)
    rmd_yr = params.start_year + (73 - params.current_age)

    milestones = [
        (retirement_yr, "Retirement", "red"),
        (ss_yr, "Social Security", "green"),
        (penalty_free_yr, "Penalty Free (60)", "orange"),
        (rmd_yr, "RMDs (73)", "purple")
    ]

    # Add Vertical Lines
    for yr, label, color in milestones:
        if df_results[x_axis].min() <= yr <= df_results[x_axis].max():
            fig.add_vline(
                x=yr, 
                line_width=1.5, 
                line_dash="dot", 
                line_color=color,
                annotation_text=label,
                annotation_position="top left",
                annotation_textangle=-90
            )
            
    # Add Abroad Regions (Background Shading)
    for start, end in params.abroad_years:
        sim_start = df_results[x_axis].min()
        sim_end = df_results[x_axis].max()
        shade_start = max(start, sim_start)
        shade_end = min(end, sim_end)
        
        if shade_start <= shade_end:
            fig.add_vrect(
                x0=shade_start, x1=shade_end,
                fillcolor="LightSalmon", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="Living Abroad", annotation_position="top left"
            )

def plot_income_and_expense_breakdown(df_results, params):
    """
    Generates detailed stacked area plots for sources of income 
    and breakdown of expenses/cash outflows over the simulation period.
    """
    print("Generating Income and Expense Breakdown Plots...")
    x_axis = 'calendar_year'

    # --- 1. Income Sources Breakdown ---
    fig_income = go.Figure()

    income_mapping = [
        ('salary_income', 'Salary'),
        ('social_security_income', 'Social Security'),
        ('strategic_pretax_withdrawal', 'Pretax Strategic/RMD'),
        ('withdrawal_pretax_extra', 'Pretax Extra Withdrawal'),
        ('withdrawal_roth', 'Roth Withdrawal'),
        ('withdrawal_cash', 'Cash/Taxable Withdrawal')
    ]

    for col, label in income_mapping:
        if col in df_results.columns:
            fig_income.add_trace(go.Scatter(
                x=df_results[x_axis],
                y=df_results[col],
                mode='none',
                stackgroup='income',
                name=label
            ))
    
    if 'rmd_amount' in df_results.columns:
        fig_income.add_trace(go.Scatter(
            x=df_results[x_axis],
            y=df_results['rmd_amount'],
            mode='lines',
            name='RMD Requirement (Ref)',
            line=dict(color='black', width=2, dash='dash')
        ))

    add_milestones_to_fig(fig_income, df_results, params)
    fig_income.update_layout(
        title="Sources of Annual Cash Flow (Income & Withdrawals)",
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig_income.show()

    # --- 2. Expense & Outflow Breakdown ---
    fig_outflow = go.Figure()

    outflow_mapping = [
        ('mortgage_payment', 'Mortgage Payment'),
        ('property_tax_payment', 'Property Tax'),
        ('healthcare_cost', 'Healthcare (Base/LTC)'), # New Field
        ('expenses', 'Living Expenses'),
        ('irmaa_cost', 'Medicare IRMAA'),
        ('tax_ordinary', 'Ordinary Income Tax'),
        ('tax_ltcg', 'Capital Gains Tax'),
        ('tax_niit', 'NIIT Tax'),
        ('tax_state', 'State Tax (NJ)'),
        ('roth_conversion_amount', 'Roth Conversion'),
        ('surplus_saved', 'Surplus Re-invested')
    ]

    for col, label in outflow_mapping:
        if col in df_results.columns:
            fig_outflow.add_trace(go.Scatter(
                x=df_results[x_axis],
                y=df_results[col],
                mode='none',
                stackgroup='outflow',
                name=label
            ))

    add_milestones_to_fig(fig_outflow, df_results, params)
    fig_outflow.update_layout(
        title="Annual Cash Outflows (Expenses, Housing, Health, Taxes & Savings)",
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig_outflow.show()

def plot_simulation_results(df_results, params):
    """
    Visualizes simulation time-series results including portfolio balances, 
    detailed income/outflow breakdowns (delegated), and tax rate analysis.
    """
    print("Generating Comprehensive Simulation Result Plots...")
    x_axis = 'calendar_year'
    
    # 1. Balance Plot (Stacked Area)
    fig_balance = df_results.plot(
        x=x_axis, 
        y=['end_pretax', 'end_roth', 'end_cash'], 
        kind='area', 
        title='Portfolio Balance Over Time'
    )
    add_milestones_to_fig(fig_balance, df_results, params)
    fig_balance.update_layout(
        xaxis_title="Year",
        yaxis_title="Balance ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig_balance.show()

    # 2 & 3. Income and Expense Breakdowns
    plot_income_and_expense_breakdown(df_results, params)

    # 4. Specific Tax Rates by Income Type (Lines)
    fig_rates = go.Figure()
    
    # Calculate component rates safely
    total_tax_safe = df_results['taxes_total'].replace(0, 1)
    
    # Bases for Rate Calculation
    taxable_salary = (df_results['salary_income'] - df_results['contribution_pretax']).clip(lower=0)
    base_cash_flow = taxable_salary + df_results['social_security_income'] - df_results['contribution_roth']
    total_tax_and_fees = df_results['taxes_total'] + df_results['irmaa_cost']
    total_outflow = df_results['expenses'] + total_tax_and_fees
    
    shortfall = (total_outflow - base_cash_flow).clip(lower=0)
    spent_forced = np.minimum(shortfall, df_results['strategic_pretax_withdrawal'])
    
    ord_income_base = (
        taxable_salary + 
        df_results['social_security_taxable'] + 
        df_results['withdrawal_pretax_extra'] + 
        df_results['roth_conversion_amount'] + 
        spent_forced
    ).replace(0, 1)

    ltcg_base = df_results['capital_gains_realized'].replace(0, 1)
    magi_base = df_results['magi'].replace(0, 1)
    
    nj_base = (
        taxable_salary + 
        ltcg_base + 
        df_results['strategic_pretax_withdrawal'] + 
        df_results['withdrawal_pretax_extra'] + 
        df_results['roth_conversion_amount']
    ).replace(0, 1)

    # Calculate Rates
    rate_ord = df_results['tax_ordinary'] / ord_income_base
    rate_ltcg = df_results['tax_ltcg'] / ltcg_base
    rate_niit = df_results['tax_niit'] / ltcg_base 
    rate_irmaa = df_results['irmaa_cost'] / magi_base
    rate_state = df_results['tax_state'] / nj_base

    rate_data = [
        (rate_ord, 'Ordinary Income Tax Rate', 'blue'),
        (rate_ltcg, 'LTCG Tax Rate', 'orange'),
        (rate_niit, 'NIIT Rate (on Gains)', 'green'),
        (rate_irmaa, 'IRMAA Effective Rate', 'red'),
        (rate_state, 'State Tax Rate (NJ)', 'purple')
    ]

    for rate_series, label, color in rate_data:
        fig_rates.add_trace(go.Scatter(
            x=df_results[x_axis],
            y=rate_series,
            name=label,
            line=dict(color=color, width=2),
            mode='lines'
        ))

    fig_rates.add_trace(go.Scatter(
        x=df_results[x_axis], 
        y=df_results['effective_tax_rate'], 
        name='Total Effective Tax Rate (excl IRMAA)',
        line=dict(color='black', width=3, dash='solid')
    ))
    
    add_milestones_to_fig(fig_rates, df_results, params)
    fig_rates.update_layout(
        title="Specific Tax Rates by Income Type (%)",
        xaxis_title="Year",
        yaxis_title="Effective Rate Contribution",
        yaxis_tickformat=".2%",
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5)
    )
    fig_rates.show()

    # 5. Tax Breakdown (Stacked Area) - Amounts Only
    fig_tax_amounts = go.Figure()
    
    tax_cols = ['tax_ordinary', 'tax_ltcg', 'tax_niit', 'tax_state']
    for col in tax_cols:
        if col in df_results.columns:
            fig_tax_amounts.add_trace(
                go.Scatter(
                    x=df_results[x_axis], 
                    y=df_results[col], 
                    name=col.replace('_', ' ').title() + " ($)",
                    stackgroup='tax_stack', 
                    mode='none'
                )
            )
        
    add_milestones_to_fig(fig_tax_amounts, df_results, params)
    fig_tax_amounts.update_layout(
        title="Annual Tax Amounts Breakdown ($)",
        xaxis_title="Year",
        yaxis_title="Tax Amount ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5)
    )
    fig_tax_amounts.show()

def plot_ordinary_tax():
    """Diagnostic plot for Ordinary Income Tax brackets."""
    print("Generating Combined Ordinary Tax Analysis...")
    incomes = np.linspace(0, 1000000, 500)
    calc = retirement_simulation.TaxCalculator
    std_deduction = retirement_simulation.TAX_CONFIG["STANDARD_DEDUCTION"]
    
    taxes = [calc.calculate_ordinary_tax(x) for x in incomes]
    eff_rates = [t/x if x > 0 else 0 for t, x in zip(taxes, incomes)]
    
    marg_rates = []
    delta = 1.0
    for x in incomes:
        t_now = calc.calculate_ordinary_tax(x)
        t_next = calc.calculate_ordinary_tax(x + delta)
        marg_rates.append((t_next - t_now) / delta)
    
    df = pd.DataFrame({
        'Ordinary Income': incomes,
        'Total Tax ($)': taxes,
        'Effective Rate': eff_rates,
        'Marginal Rate': marg_rates
    })

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Ordinary Income'], y=df['Effective Rate'], name='Effective Rate'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Ordinary Income'], y=df['Marginal Rate'], name='Marginal Rate'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Ordinary Income'], y=df['Total Tax ($)'], name='Total Tax ($)', line=dict(dash='dash')), secondary_y=True)

    fig.update_layout(
        title=f"Ordinary Income Tax Analysis (Std Deduction: ${std_deduction:,})",
        xaxis_title="Ordinary Income ($)",
        yaxis_title="Tax Rate",
        yaxis_tickformat=".1%",
        yaxis2_title="Total Tax ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig.show()

def plot_ltcg_stacking():
    """Diagnostic plot for LTCG Tax Stacking."""
    print("Generating Combined LTCG Stacking Analysis...")
    ordinary_bases = [0, 50000, 200000]
    ltcg_range = np.linspace(0, 600000, 500)
    calc = retirement_simulation.TaxCalculator
    std_deduction = retirement_simulation.TAX_CONFIG["STANDARD_DEDUCTION"]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = px.colors.qualitative.Plotly
    
    for i, ord_base in enumerate(ordinary_bases):
        taxable_ord_base = max(0, ord_base - std_deduction)
        taxes, marg_rates = [], []
        delta = 1.0
        
        for gain in ltcg_range:
            t = calc.calculate_ltcg_tax(taxable_ord_base, gain)
            taxes.append(t)
            t_next = calc.calculate_ltcg_tax(taxable_ord_base, gain + delta)
            marg_rates.append((t_next - t) / delta)
        
        color = colors[i % len(colors)]
        label = f'Base Ord: ${ord_base:,}'
        fig.add_trace(go.Scatter(x=ltcg_range, y=marg_rates, name=f'{label} (Marginal Rate)', line=dict(color=color)), secondary_y=False)
        fig.add_trace(go.Scatter(x=ltcg_range, y=taxes, name=f'{label} (Total Tax)', line=dict(color=color, dash='dot')), secondary_y=True)

    fig.update_layout(
        title="LTCG Tax Liability and Marginal Rates (Tax Stacking)",
        xaxis_title="Long Term Capital Gains Amount ($)",
        yaxis_title="Marginal Rate",
        yaxis_tickformat=".1%",
        yaxis2_title="LTCG Tax ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig.show()

def plot_niit_behavior():
    """Diagnostic plot for NIIT."""
    print("Generating Combined NIIT Analysis...")
    calc = retirement_simulation.TaxCalculator
    niit_threshold = retirement_simulation.TAX_CONFIG["NIIT_THRESHOLD"]
    ordinary_bases = [0, 50000, 200000]
    ltcg_range = np.linspace(0, 600000, 500)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = px.colors.qualitative.Safe

    for i, ord_base in enumerate(ordinary_bases):
        taxes, marg_rates = [], []
        delta = 1.0
        for gain in ltcg_range:
            t = calc.calculate_detailed_tax(ord_base, gain, 0, 0, 0, 65)["niit"]
            taxes.append(t)
            t_next = calc.calculate_detailed_tax(ord_base, gain + delta, 0, 0, 0, 65)["niit"]
            marg_rates.append((t_next - t) / delta)
            
        color = colors[i % len(colors)]
        label = f'Base Ord: ${ord_base:,}'
        fig.add_trace(go.Scatter(x=ltcg_range, y=marg_rates, name=f'{label} (NIIT Marg Rate)', line=dict(color=color)), secondary_y=False)
        fig.add_trace(go.Scatter(x=ltcg_range, y=taxes, name=f'{label} (NIIT Tax)', line=dict(color=color, dash='dash')), secondary_y=True)

    fig.update_layout(
        title=f"Net Investment Income Tax Analysis (Threshold: ${niit_threshold:,})",
        xaxis_title="Long Term Capital Gains ($)",
        yaxis_title="NIIT Marginal Rate",
        yaxis_tickformat=".1%",
        yaxis2_title="NIIT Tax Amount ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig.show()

def plot_irmaa_behavior():
    """Diagnostic plot for Medicare IRMAA Surcharges (Cliffs)."""
    print("Generating Combined IRMAA Analysis...")
    calc = retirement_simulation.TaxCalculator
    
    magi_range = np.linspace(150000, 1000000, 2000)
    surcharges, marg_rates = [], []
    delta = 1.0 
    
    for magi in magi_range:
        cost = calc.calculate_irmaa_surcharge(magi)
        surcharges.append(cost)
        cost_next = calc.calculate_irmaa_surcharge(magi + delta)
        marg_rates.append((cost_next - cost) / delta)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=magi_range, y=marg_rates, name='Marginal Rate (Cliff Impact)', line=dict(color='red', width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=magi_range, y=surcharges, name='Annual IRMAA Surcharge (Couple)', line=dict(color='blue', dash='solid', width=3)), secondary_y=True)

    brackets = retirement_simulation.TAX_CONFIG.get("IRMAA_BRACKETS", [])
    for threshold, _, _ in brackets:
        if threshold != float('inf'):
            fig.add_vline(x=threshold, line_width=1, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Medicare IRMAA Surcharge Analysis (Cliffs)",
        xaxis_title="Modified Adjusted Gross Income (MAGI) ($)",
        yaxis_title="Marginal Rate (Cliff Spike)",
        yaxis_tickformat=".0%", 
        yaxis2_title="Total Annual Surcharge ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig.show()

def plot_nj_state_tax_behavior():
    """Diagnostic plot for NJ State Tax with Retirement Exclusions."""
    print("Generating NJ State Tax Analysis...")
    calc = retirement_simulation.TaxCalculator
    
    incomes = np.linspace(0, 300000, 1000)
    ages = [60, 65]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = px.colors.qualitative.Plotly
    
    for i, age in enumerate(ages):
        taxes, eff_rates, marg_rates = [], [], []
        delta = 100.0 
        
        for income in incomes:
            t = calc.calculate_nj_tax(salary=0, pension_income=income, capital_gains=0, age=age)
            taxes.append(t)
            eff_rates.append(t / income if income > 0 else 0)
            t_next = calc.calculate_nj_tax(salary=0, pension_income=income+delta, capital_gains=0, age=age)
            marg_rates.append((t_next - t) / delta)
            
        color = colors[i]
        label = f"Age {age} {'(Retirement Exclusion)' if age >= 62 else '(Standard)'}"
        fig.add_trace(go.Scatter(x=incomes, y=eff_rates, name=f'{label} Eff Rate', line=dict(color=color, width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=incomes, y=taxes, name=f'{label} Total Tax', line=dict(color=color, dash='dot')), secondary_y=True)

    fig.add_vline(x=100000, line_width=1, line_dash="dot", line_color="gray", annotation_text="100k Limit")
    fig.add_vline(x=150000, line_width=1, line_dash="dot", line_color="gray", annotation_text="150k Cliff")

    fig.update_layout(
        title="NJ State Tax Analysis (Retirement Exclusion Cliffs)",
        xaxis_title="Gross Income ($)",
        yaxis_title="Effective Tax Rate",
        yaxis_tickformat=".1%", 
        yaxis2_title="Total State Tax ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    fig.show()

if __name__ == "__main__":
    try:
        plot_ordinary_tax()
        plot_ltcg_stacking()
        plot_niit_behavior()
        plot_irmaa_behavior()
        plot_nj_state_tax_behavior()
        
        current_yr = datetime.now().year
        params = retirement_simulation.SimParameters(
            current_age=40, retirement_age=65, death_age=90, start_year=current_yr,
            cash_savings=7e6, taxable_account_basis=0, pretax_balance=1e6, roth_balance=5.8e5,
            retirement_return_rate=0.06, taxable_return_rate=0.04, inflation_rate=0.03,
            annual_salary=5e5, annual_pretax_contribution=7e4, annual_roth_contribution=4.5e4,
            social_security=25000, social_security_start_age=67,
            default_expense_params=retirement_simulation.ExpenseParams(
                method='asset_ratio', base_annual_expense=3e5, expense_ratio=0.04, max_annual_expense=5e5
            ),
            annual_pretax_withdrawal_ratio=0.02,
            do_roth_conversions=True, conversion_start_age=65, conversion_end_age=59,
            annual_conversion_amount=20000
        )
        
        df_results = retirement_simulation.run_simulation(params)
        plot_simulation_results(df_results, params)
        
        print("All plots generated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}\nEnsure retirement_simulation.py is in your path.")
