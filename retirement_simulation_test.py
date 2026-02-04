import unittest
from retirement_simulation import (
    SimParameters, 
    YearData, 
    calculate_detailed_taxes, 
    calculate_year, 
    run_simulation,
    ORDINARY_BRACKETS,
    LTCG_BRACKETS
)

class TestTaxCalculations(unittest.TestCase):
    def setUp(self):
        self.std_deduction = 14600
        self.niit_threshold = 200000

    def test_standard_deduction_only(self):
        """Income below standard deduction should result in 0 tax."""
        tax = calculate_detailed_taxes(14000, 0, self.std_deduction, self.niit_threshold)
        self.assertEqual(tax, 0.0)

    def test_ordinary_income_simple_bracket(self):
        """Test tax within the first bracket (10%)."""
        # Income: 24,600. Taxable: 10,000. Tax: 10% of 10,000 = 1,000.
        income = 24600
        expected_tax = 1000.0
        tax = calculate_detailed_taxes(income, 0, self.std_deduction, self.niit_threshold)
        self.assertAlmostEqual(tax, expected_tax, places=2)

    def test_ltcg_stacking(self):
        """
        Test that Long Term Capital Gains stack ON TOP of Ordinary Income.
        Scenario:
        - Ordinary Taxable: 40,000 (Falls in 12% bracket)
        - LTCG: 20,000
        
        The LTCG starts at 40,000.
        LTCG 0% bracket ends at 47,025.
        
        LTCG Tax Calculation:
        - 7,025 fills the rest of the 0% bucket (47,025 - 40,000). Tax = 0.
        - Remaining 12,975 falls into 15% bucket. Tax = 12,975 * 0.15 = 1,946.25.
        
        Ordinary Tax Calculation (using 2024 brackets approx for validation):
        - 11,600 @ 10% = 1,160
        - (40,000 - 11,600) @ 12% = 28,400 * 0.12 = 3,408
        - Total Ordinary = 4,568
        
        Total Expected = 4,568 + 1,946.25 = 6,514.25
        """
        ordinary_income = 40000 + self.std_deduction # Gross income to get 40k taxable
        ltcg = 20000
        
        tax = calculate_detailed_taxes(ordinary_income, ltcg, self.std_deduction, self.niit_threshold)
        
        # Calculate expected ordinary
        taxable_ord = 40000
        expected_ord = (11600 * 0.10) + ((40000 - 11600) * 0.12)
        
        # Calculate expected LTCG
        # Room in 0% bucket (limit 47025) starting from 40000
        room_0pct = max(0, 47025 - 40000) # 7025
        taxable_0pct = min(ltcg, room_0pct)
        taxable_15pct = ltcg - taxable_0pct
        expected_ltcg = (taxable_0pct * 0.0) + (taxable_15pct * 0.15)
        
        self.assertAlmostEqual(tax, expected_ord + expected_ltcg, places=2)

    def test_niit_trigger(self):
        """Test Net Investment Income Tax trigger."""
        # Ordinary: 200,000 (Taxable approx 185,400)
        # LTCG: 50,000
        # MAGI = 250,000. Threshold = 200,000. Excess = 50,000.
        # NIIT = 3.8% on min(LTCG, Excess) = 3.8% on 50,000 = 1,900.
        
        ordinary = 200000
        ltcg = 50000
        
        # Get base tax without NIIT manually or via trusted path? 
        # We'll just verify the total tax is higher than standard calc by exactly the NIIT amount.
        tax_with_niit = calculate_detailed_taxes(ordinary, ltcg, 0, 200000) # 0 deduction for simplicity
        tax_without_niit_logic = calculate_detailed_taxes(ordinary, ltcg, 0, 1000000) # High threshold
        
        niit_amount = tax_with_niit - tax_without_niit_logic
        self.assertAlmostEqual(niit_amount, 50000 * 0.038, places=2)


class TestYearlySimulationLogic(unittest.TestCase):
    def setUp(self):
        # Base parameters for a generic year
        self.params = SimParameters(
            current_age=50, retirement_age=60, death_age=90,
            cash_savings=100000, taxable_account_basis=100000, # No unrealized gains initially
            pretax_balance=100000, roth_balance=100000,
            return_rate=0.0, inflation_rate=0.0, # Zero growth/inflation for easy logic checks
            annual_salary=0, social_security=0, social_security_start_age=70,
            living_expense_method='fixed_inflation', base_annual_expense=50000, expense_ratio=0.0,
            do_roth_conversions=False, conversion_start_age=0, conversion_end_age=0, annual_conversion_amount=0,
            standard_deduction=0, niit_threshold=1000000
        )
        
        self.start_year = YearData(
            age=50, year_index=0,
            start_pretax=0, start_roth=0, start_cash=0, start_taxable_basis=0, start_total=0,
            salary_income=0, social_security_income=0,
            growth_pretax=0, growth_roth=0, growth_cash=0,
            expenses=0, roth_conversion_amount=0,
            withdrawal_pretax=0, withdrawal_roth=0, withdrawal_cash=0, capital_gains_realized=0,
            taxes_paid=0, effective_tax_rate=0,
            end_pretax=100000, end_roth=100000, end_cash=100000, end_taxable_basis=100000, end_total=300000
        )

    def test_withdrawal_order(self):
        """
        Verify order: Income -> Cash -> Roth -> Pretax.
        Expense: 50k.
        Salary: 0.
        Cash: 100k.
        Should take 50k from Cash. Roth and Pretax untouched.
        """
        self.params.base_annual_expense = 50000
        result = calculate_year(self.start_year, self.params)
        
        self.assertEqual(result.withdrawal_cash, 50000)
        self.assertEqual(result.withdrawal_roth, 0)
        self.assertEqual(result.withdrawal_pretax, 0)
        self.assertEqual(result.end_cash, 50000)

    def test_withdrawal_spillover_to_roth(self):
        """
        Expense: 150k.
        Cash: 100k.
        Roth: 100k.
        Should take 100k Cash, then 50k Roth.
        """
        self.params.base_annual_expense = 150000
        result = calculate_year(self.start_year, self.params)
        
        self.assertEqual(result.withdrawal_cash, 100000)
        self.assertEqual(result.withdrawal_roth, 50000)
        self.assertEqual(result.withdrawal_pretax, 0)
        
        self.assertEqual(result.end_cash, 0)
        self.assertEqual(result.end_roth, 50000)

    def test_roth_conversion_logic(self):
        """
        Enable conversions.
        Age 51 (Start year age 50 + 1).
        Conversion Amount: 10k.
        Should move 10k from Pretax to Roth.
        """
        self.params.do_roth_conversions = True
        self.params.conversion_start_age = 50
        self.params.conversion_end_age = 60
        self.params.annual_conversion_amount = 10000
        self.params.base_annual_expense = 0 # No expenses to confuse things
        
        result = calculate_year(self.start_year, self.params)
        
        self.assertEqual(result.roth_conversion_amount, 10000)
        # Pretax started 100k -> minus 10k conversion = 90k
        self.assertEqual(result.end_pretax, 90000) 
        # Roth started 100k -> plus 10k conversion = 110k
        self.assertEqual(result.end_roth, 110000)
        
        # Check that taxes were paid on conversion
        # 10k income @ 0 deduction. 10% bracket. Tax ~1000.
        self.assertTrue(result.taxes_paid > 0)

    def test_taxable_basis_calculation(self):
        """
        Test that withdrawing from a taxable account with gains triggers realized gains
        and reduces basis proportionally.
        
        Start Cash: 100k.
        Start Basis: 50k. (50% is gains).
        Withdrawal: 10k.
        
        Realized Gain should be 50% of 10k = 5k.
        Basis should reduce by 50% of 10k = 5k.
        New Basis = 45k.
        """
        self.start_year.end_taxable_basis = 50000 # Override basis
        self.params.base_annual_expense = 10000
        
        result = calculate_year(self.start_year, self.params)
        
        self.assertEqual(result.withdrawal_cash, 10000)
        self.assertAlmostEqual(result.capital_gains_realized, 5000)
        self.assertAlmostEqual(result.end_taxable_basis, 45000)

class TestFullIntegration(unittest.TestCase):
    def test_run_simulation_completes(self):
        """Ensure the main loop runs and produces a DataFrame."""
        params = SimParameters(
            current_age=60, retirement_age=65, death_age=70,
            cash_savings=10000, taxable_account_basis=10000,
            pretax_balance=100000, roth_balance=10000,
            return_rate=0.05, inflation_rate=0.03,
            annual_salary=50000, social_security=20000, social_security_start_age=67,
            living_expense_method='fixed_inflation', base_annual_expense=40000, expense_ratio=0.0,
            do_roth_conversions=True, conversion_start_age=60, conversion_end_age=65, annual_conversion_amount=5000,
            standard_deduction=14600, niit_threshold=200000
        )
        
        df = run_simulation(params)
        
        # 60 to 70 is 10 years (or 11 rows including start state)
        self.assertTrue(len(df) >= 10)
        self.assertIn('end_total', df.columns)
        self.assertIn('taxes_paid', df.columns)

if __name__ == '__main__':
    unittest.main()
