## Note - May need to update the top level tax parameters.

Retirement Simulation Methodology

This document outlines the step-by-step mathematical logic and financial rules used in the retirement_simulation.py script.

1. Timeline and Economic Core

The simulation progresses year-by-year starting from the start_year until the death_age or until total assets reach zero. The tax parameters are based on the 2025 Tax Year.

Inflation Compounding

Inflation is compounded annually. For any given year $t$:

$$\text{Inflation Factor}_t = \prod_{i=0}^{t} (1 + \text{Inflation Rate}_i)$$

All income (Salary, Social Security) and variable expenses (Lifestyle, Property Tax, Healthcare) are multiplied by this factor to maintain "today's dollars" purchasing power.

Overrides: Specific years can have custom inflation rates defined in return_overrides.

Asset Growth

Assets are grown at the start of each year before any withdrawals or contributions occur.

Retirement Accounts: $\text{Balance}_{new} = \text{Balance}_{old} \times (1 + \text{Retirement Return Rate})$

Taxable Accounts: $\text{Balance}_{new} = \text{Balance}_{old} \times (1 + \text{Taxable Return Rate})$

Overrides: Specific years can have custom return rates to simulate market crashes or booms.

2. Income and Contributions

Salary Logic & Living Abroad

Base Salary: Defined by annual_salary, adjustable via salary_overrides for specific periods (e.g., consulting, sabbaticals).

Taxable Salary Calculation: $\text{Gross Salary} - \text{Pretax Contribution}$.

Living Abroad (FEIE): During years marked as "abroad":

Federal: The Foreign Earned Income Exclusion (FEIE) reduces taxable salary by the inflation-adjusted limit (~$130k for 2025). Passive income (pension, dividends) remains fully taxable.

State: NJ State Income Tax is assumed to be $0 on all income sources due to non-residency.

Social Security Taxation (Federal)

The simulation calculates the taxable portion of Social Security based on "Provisional Income" (Combined Income Rule):

$$\text{Provisional Income} = \text{Other Taxable Income} + 50\% \text{ of SS Benefit}$$

For Married Filing Jointly (MFJ):

If $<\$32k$: $0\%$ taxable.

If $\$32k$ to $\$44k$: Up to $50\%$ taxable.

If $>\$44k$: Up to $85\%$ taxable.

3. Expenses and Liabilities

The simulation aggregates multiple expense categories into a single Total Cash Outflow Target.

A. Lifestyle Expenses

Calculated using one of two methods, configurable by year range:

Fixed Inflation: $\text{Base Amount} \times \text{Inflation Factor}$.

Asset Ratio: $\text{Total Portfolio} \times \text{Expense Ratio}$, capped at max_annual_expense (inflation-adjusted).

B. Housing Costs

Mortgage: Treated as a fixed nominal cost (no inflation). It applies only for the duration of mortgage_remaining_years.

Property Tax: Treated as a variable cost. $\text{Base Amount} \times \text{Inflation Factor}$.

C. Healthcare Costs

Defined by specific age ranges to model the three phases of retirement health spending:

Gap Years (Pre-65): High cost (ACA/Private insurance).

Medicare Years (65–85): Moderate cost (Medicare Part B + Medigap + Part D).

Late Stage (85+): High cost (Long-Term Care risk).

Note: These base costs are inflated annually.

D. Medicare IRMAA Surcharges

If Age $\ge 65$, an income-related surcharge is added on top of the base healthcare cost based on the MAGI from 2 years prior.

Cliff Logic: Exceeding a threshold by $1 triggers the full surcharge for that tier.

Tiers: Based on 2025 MFJ brackets (e.g., surcharge starts at MAGI > $212k$).

4. Forced and Strategic Withdrawals

Every year, a "Forced Pretax Withdrawal" is calculated as:

$$\text{Forced Withdrawal} = \max(\text{RMD Amount}, \text{Strategic Ratio Amount})$$

RMD: Based on the IRS Uniform Lifetime Table starting at age 73.

Strategic Ratio: $\text{Total Portfolio Value} \times \text{Annual Pretax Withdrawal Ratio}$ (starting at age 60).

The "Roth Spillover" Rule: Any portion of this forced withdrawal that is not needed to cover expenses/taxes is automatically converted into the Roth account.

5. The Withdrawal Hierarchy (Order of Operations)

To cover the sum of Lifestyle + Housing + Healthcare + Taxes + IRMAA, the simulation follows this priority:

Net Income: Salary (post-pretax) + Social Security.

Taxable Cash Savings: Drawing from the brokerage account.

Forced Pretax Distributions: Using the RMD/Strategic amount calculated in Step 4.

Roth Assets: Tax-free withdrawals from the Roth account.

Extra Pretax Withdrawals: Additional 401k/IRA withdrawals (only if the above are exhausted).

6. Tax Calculation

The simulation uses an iterative solver (3–5 passes) to handle circular logic (withdrawals causing taxes causing more withdrawals).

Federal Tax Categories (2025 Rates)

Ordinary Tax: Progressive brackets ($10\%$ to $37\%$) applied to Taxable Salary (post-FEIE), Taxable SS, Pretax withdrawals, and RMDs.

LTCG Tax: $0\%$, $15\%$, or $20\%$ based on total taxable income, applied to realized gains from the taxable account.

NIIT: $3.8\%$ tax on investment income if MAGI exceeds $\$250,000$.

State Tax (New Jersey)

NJ Tax is calculated on "Gross Income" with specific rules:

Social Security: $100\%$ Exempt.

Capital Gains: Taxed at ordinary state rates (no preferential rate).

Residency: Tax is $0 if living abroad.

Pension Exclusion (Age 62+):

Income $\le \$100k$: Exclude up to $\$100k$.

Income $\le \$125k$: Exclude $50\%$ of pension/annuity income.

Income $\le \$150k$: Exclude $25\%$ of pension/annuity income.

Income $> \$150k$: Cliff! $0\%$ exclusion.

7. Basis Tracking

The simulation tracks the "Taxable Basis" of the cash account. When funds are withdrawn from the taxable account:

$$\text{Realized Gain} = \text{Withdrawal} \times \frac{\text{Balance} - \text{Basis}}{\text{Balance}}$$

The basis is then reduced proportionally to the withdrawal to ensure future gain calculations are accurate.

8. Unsupported Areas & Future Improvements

While the simulation handles complex tax interactions, the following financial aspects are currently not modeled or are simplified:

Spousal & Survivorship Dynamics

The "Widow(er) Tax Penalty": The simulation assumes "Married Filing Jointly" status until death_age. It does not model the transition to "Single" filing status upon the death of a spouse, which typically pushes the survivor into higher tax brackets with less income.

Survivor Benefits: Social Security income remains constant (inflation-adjusted). The loss of the smaller Social Security check upon one spouse's death is not simulated.

Real Estate & Illiquid Assets

Home Equity: The simulation tracks only liquid investment assets (Cash, Pretax, Roth). It does not account for home equity, property appreciation, or the cash flow impact of downsizing or reverse mortgages.

Tax Deductibility: Mortgage interest and Property Taxes are treated purely as cash outflows. They are not currently itemized for tax deductions (Standard Deduction is assumed).

Advanced Healthcare Logic

Health Inflation: All expenses grow at the general CPI rate. Historically, healthcare costs have risen faster than general inflation.

Market Stochastics (Monte Carlo)

Sequence of Returns: The simulation is deterministic (one fixed path). While manual overrides can simulate a crash, it does not calculate probability of success (e.g., "95% chance of not running out of money") across thousands of random market scenarios.

Dynamic Withdrawal Rules

Guardrails (Guyton-Klinger): The current model calculates spending based on static rules. It does not support dynamic adjustments found in advanced "Guardrails" strategies, such as:

Capital Preservation Rule: If the portfolio performs poorly and the withdrawal rate rises 20% above the initial target (e.g., from 4% to 4.8%), spending is cut by 10% to prevent depletion.

Prosperity Rule: If the portfolio performs well and the withdrawal rate drops 20% below the target (e.g., from 4% to 3.2%), spending is increased by 10% to avoid dying with too much wealth.

Inflation Skip: Skipping the annual inflation adjustment following a year with negative portfolio returns.
