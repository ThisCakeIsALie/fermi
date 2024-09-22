import sys
import re
import numpy as np
from scipy.stats import truncnorm
import argparse

def parse_expression(expr):
    number_pattern = r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
    range_pattern = rf'({number_pattern})\s*,\s*({number_pattern})'
    token_pattern = rf'({range_pattern}|{number_pattern})'

    tokens = []
    var_index = 0

    def replace_token(match):
        nonlocal var_index
        token = match.group(0)
        var_name = f'var{var_index}'
        var_index += 1
        tokens.append((var_name, token))
        return var_name

    # Replace numbers and ranges with variable names
    expression = re.sub(token_pattern, replace_token, expr)

    # Build variables dictionary
    var_values = {}
    for var_name, token in tokens:
        # Check if token is a range
        range_match = re.fullmatch(range_pattern, token)
        if range_match:
            a = float(range_match.group(1))
            b = float(range_match.group(2))
            if a > b:
                print(f"fermi: error: lower bound {a} is greater than upper bound {b} in range '{token}'")
                sys.exit(1)
            var_values[var_name] = (a, b)
        else:
            var_values[var_name] = float(token)

    if ',' in expression:
        print(f"fermi: error: invalid character ',' in expression. Perhaps you provided a range with too many values?")
        sys.exit(1)

    return expression, var_values

def generate_samples(var_values, num_samples):
    samples = {}
    for var_name, value in var_values.items():
        if isinstance(value, tuple):
            a, b = value
            if a == b:
                samples[var_name] = np.full(num_samples, a)
            elif a > 0:
                # Use log-normal distribution
                sigma = (np.log(b) - np.log(a)) / (2 * 2.576)
                if sigma <= 0:
                    print(f"fermi: error: invalid range for log-normal distribution in variable '{var_name}'")
                    sys.exit(1)
                mu = (np.log(a) + np.log(b)) / 2
                samples[var_name] = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
            else:
                # Use truncated normal distribution
                mu = (a + b) / 2
                sigma = (b - a) / (2 * 2.576)
                if sigma <= 0:
                    print(f"fermi: error: invalid range for normal distribution in variable '{var_name}'")
                    sys.exit(1)
                lower, upper = (a - mu) / sigma, (b - mu) / sigma
                samples[var_name] = truncnorm.rvs(lower, upper, loc=mu, scale=sigma, size=num_samples)
        else:
            # Single value
            samples[var_name] = np.full(num_samples, value)
    return samples

def evaluate_expression(expression, samples):
    # Safe dictionary for eval
    allowed_names = {
        'np': np,
        '__builtins__': {}
    }
    allowed_names.update(samples)

    # Replace '^' with '**' for exponentiation
    expression = expression.replace('^', '**')

    # Evaluate the expression
    try:
        result = eval(expression, allowed_names)
    except Exception as e:
        print(f"fermi: error: Error evaluating expression: {e}")
        sys.exit(1)
    return result

def unsafe_main():
    parser = argparse.ArgumentParser(description='Fermi estimation using Monte Carlo simulation.')
    parser.add_argument('expression', help='Arithmetic expression for estimation.')
    parser.add_argument('-n', '--samples', type=int, default=10000, help='Number of Monte Carlo samples.')
    parser.add_argument('-m', '--mean', action='store_true', help='Output the mean of the estimate.')
    parser.add_argument('-s', '--std', action='store_true', help='Output the standard deviation of the estimate.')
    parser.add_argument('-c', '--ci', action='store_true', help='Output the 99%% confidence interval of the estimate.')
    args = parser.parse_args()
    args = parser.parse_args()

    expr = args.expression
    num_samples = args.samples

    expression, var_values = parse_expression(expr)
    samples = generate_samples(var_values, num_samples)
    result = evaluate_expression(expression, samples)

    if not isinstance(result, np.ndarray):
        print(f"fermi: error: expression '{expr}' must evaluate to a NumPy array")
        sys.exit(1)

    # Compute statistics
    mean = np.mean(result)
    std_dev = np.std(result)
    ci_lower = np.percentile(result, 1)
    ci_upper = np.percentile(result, 99)

    outputs = []
    if args.mean or args.std or args.ci:
        if args.mean:
            outputs.append(f"{mean:.2f}")
        if args.std:
            outputs.append(f"{std_dev:.2f}")
        if args.ci:
            outputs.append(f"{ci_lower:.2f},{ci_upper:.2f}")
    else:
        outputs.append(f"{mean:.2f} {std_dev:.2f} {ci_lower:.2f},{ci_upper:.2f}")

    print(' '.join(outputs))

def main():
    try:
        unsafe_main()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"fermi: error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()