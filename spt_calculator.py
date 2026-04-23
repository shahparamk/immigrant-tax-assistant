"""
Substantial Presence Test Calculator
Pure Python — deterministic, no LLM involved.
Per IRS Publication 519, Chapter 1, Tax Year 2024
"""

def calculate_spt(days_current_year: int, days_year_minus_1: int, days_year_minus_2: int) -> dict:
    """
    Calculate Substantial Presence Test result.
    
    Formula:
        All days current year
      + 1/3 days in year-1
      + 1/6 days in year-2
      >= 183 AND >= 31 days current year → Resident Alien
    
    Args:
        days_current_year: Days present in current tax year
        days_year_minus_1: Days present in prior year
        days_year_minus_2: Days present two years ago
    
    Returns:
        dict with result, explanation, and filing guidance
    """
    weighted_y1 = days_year_minus_1 / 3
    weighted_y2 = days_year_minus_2 / 6
    total_testing_days = days_current_year + weighted_y1 + weighted_y2

    meets_31_day = days_current_year >= 31
    meets_183_day = total_testing_days >= 183
    passes_spt = meets_31_day and meets_183_day

    return {
        'days_current_year': days_current_year,
        'days_year_minus_1': days_year_minus_1,
        'days_year_minus_2': days_year_minus_2,
        'weighted_year_minus_1': round(weighted_y1, 2),
        'weighted_year_minus_2': round(weighted_y2, 2),
        'total_testing_days': round(total_testing_days, 2),
        'meets_31_day_requirement': meets_31_day,
        'meets_183_day_requirement': meets_183_day,
        'passes_spt': passes_spt,
        'tax_status': 'Resident Alien' if passes_spt else 'Nonresident Alien',
        'form_to_file': 'Form 1040' if passes_spt else 'Form 1040-NR',
        'citation': 'IRS Publication 519, Chapter 1 — Substantial Presence Test, Tax Year 2024',
        'explanation': (
            f"Days current year: {days_current_year} × 1.000 = {days_current_year}\n"
            f"Days year-1:       {days_year_minus_1} × 0.333 = {weighted_y1:.2f}\n"
            f"Days year-2:       {days_year_minus_2} × 0.167 = {weighted_y2:.2f}\n"
            f"Total testing days: {total_testing_days:.2f}\n"
            f"Meets 31-day rule:  {'YES' if meets_31_day else 'NO'}\n"
            f"Meets 183-day rule: {'YES' if meets_183_day else 'NO'}\n"
            f"Result: {'PASSES → Resident Alien → File Form 1040' if passes_spt else 'FAILS → Nonresident Alien → File Form 1040-NR'}\n"
            f"Per: IRS Publication 519, Chapter 1, Tax Year 2024"
        )
    }


if __name__ == '__main__':
    print("=== SPT CALCULATOR TEST CASES ===\n")
    
    cases = [
        (180, 0, 0, "F-1 Student Year 1-2"),
        (300, 300, 300, "H-1B Worker Year 4+"),
        (31, 456, 0, "Edge Case — exactly 183"),
        (200, 200, 200, "OPT Worker Year 3"),
    ]
    
    for current, y1, y2, label in cases:
        result = calculate_spt(current, y1, y2)
        print(f"[{label}]")
        print(result['explanation'])
        print()
