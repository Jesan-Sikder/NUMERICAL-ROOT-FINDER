"""
Newton-Raphson Method for Root Finding
=======================================

This module implements the Newton-Raphson algorithm to find roots of
nonlinear equations. The method uses the derivative to iteratively
refine the approximation: x_{n+1} = x_n - f(x_n)/f'(x_n)

Author: Numerical Methods Project
"""

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import Tuple, List, Dict


def newton_raphson(
    f_expr: str,
    x0: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[float, List[Dict], bool]:
    """
    Perform Newton-Raphson Method to find the root of f(x) = 0.
    
    The method iterates using:
        x_{n+1} = x_n - f(x_n)/f'(x_n)
    
    The derivative f'(x) is computed automatically using SymPy.
    
    Parameters:
    -----------
    f_expr : str
        The function f(x) as a string expression (e.g., 'x**2 - 4')
    x0 : float
        Initial guess for the root
    tolerance : float
        Convergence criterion (default: 1e-6)
    max_iterations : int
        Maximum number of iterations (default: 100)
    
    Returns:
    --------
    Tuple[float, List[Dict], bool]
        - Final approximation of the root
        - List of iteration data (iteration number, x value, f(x), error)
        - Boolean indicating if method converged
    
    Example:
    --------
    >>> root, iterations, converged = newton_raphson('x**2 - 4', 1.0, 1e-6, 100)
    >>> print(f"Root: {root:.6f}")
    """
    
    # Define symbolic variable
    x = sp.Symbol('x')
    
    # Parse the function f(x)
    try:
        f = parse_expr(f_expr, transformations='all')
    except Exception as e:
        raise ValueError(f"Error parsing function: {e}")
    
    # Compute derivative f'(x) using SymPy
    f_prime = sp.diff(f, x)
    
    # Convert to numerical functions
    f_func = sp.lambdify(x, f, 'numpy')
    f_prime_func = sp.lambdify(x, f_prime, 'numpy')
    
    # Initialize iteration tracking
    iterations_data = []
    x_current = x0
    converged = False
    
    print("\n" + "="*80)
    print("NEWTON-RAPHSON METHOD")
    print("="*80)
    print(f"Function f(x): {f}")
    print(f"Derivative f'(x): {f_prime}")
    print(f"Initial guess x0: {x0}")
    print(f"Tolerance: {tolerance}")
    print(f"Maximum iterations: {max_iterations}")
    print("="*80)
    print(f"{'Iteration':<12} {'x':<20} {'f(x)':<20} {'Error':<20}")
    print("-"*80)
    
    # Iteration 0 (initial guess)
    try:
        f_current = float(f_func(x_current))
    except Exception as e:
        raise ValueError(f"Error evaluating function at initial guess: {e}")
    
    iterations_data.append({
        'iteration': 0,
        'x': x_current,
        'fx': f_current,
        'error': None
    })
    print(f"{0:<12} {x_current:<20.10f} {f_current:<20.10e} {'N/A':<20}")
    
    # Perform iterations
    for iteration in range(1, max_iterations + 1):
        # Evaluate f'(x_n)
        try:
            f_prime_current = float(f_prime_func(x_current))
        except Exception as e:
            print(f"\nError evaluating derivative at x = {x_current}: {e}")
            break
        
        # Check if derivative is zero (would cause division by zero)
        if abs(f_prime_current) < 1e-12:
            print(f"\nWarning: Derivative too close to zero at x = {x_current}")
            print("Cannot continue iteration (division by zero)")
            break
        
        # Compute next value: x_{n+1} = x_n - f(x_n)/f'(x_n)
        x_next = x_current - (f_current / f_prime_current)
        
        # Evaluate f(x_{n+1})
        try:
            f_next = float(f_func(x_next))
        except Exception as e:
            print(f"\nError evaluating function at x = {x_next}: {e}")
            break
        
        # Calculate error (absolute difference)
        error = abs(x_next - x_current)
        
        # Store iteration data
        iterations_data.append({
            'iteration': iteration,
            'x': x_next,
            'fx': f_next,
            'error': error
        })
        
        # Print iteration details
        print(f"{iteration:<12} {x_next:<20.10f} {f_next:<20.10e} {error:<20.10e}")
        
        # Check convergence (can use either error in x or |f(x)|)
        if error < tolerance or abs(f_next) < tolerance:
            converged = True
            break
        
        # Update for next iteration
        x_current = x_next
        f_current = f_next
    
    print("-"*80)
    
    if converged:
        print(f"\n✓ Converged after {iteration} iterations")
        print(f"✓ Root: x ≈ {x_next:.10f}")
        print(f"✓ f(root) ≈ {f_next:.10e}")
    else:
        print(f"\n✗ Failed to converge within {max_iterations} iterations")
        print(f"✗ Last approximation: x ≈ {x_current:.10f}")
        print(f"✗ f(x) ≈ {f_current:.10e}")
    
    print("="*80)
    
    final_root = x_next if converged else x_current
    return final_root, iterations_data, converged


def main():
    """
    Main function demonstrating the Newton-Raphson method.
    Allows user input for interactive solving.
    """
    print("\n" + "="*80)
    print("NEWTON-RAPHSON METHOD - INTERACTIVE SOLVER")
    print("="*80)
    print("\nThis program finds roots of the equation f(x) = 0")
    print("The derivative is computed automatically using SymPy")
    print()
    print("Examples:")
    print("  - x**2 - 4            (roots at x = ±2)")
    print("  - x**3 - x - 2        (polynomial)")
    print("  - cos(x) - x          (transcendental)")
    print("  - exp(x) - 2          (exponential)")
    print()
    
    # Get user input
    try:
        f_expr = input("Enter f(x) (e.g., 'x**2 - 4', 'cos(x) - x'): ").strip()
        x0 = float(input("Enter initial guess x0: "))
        tolerance = float(input("Enter tolerance (e.g., 1e-6): ") or "1e-6")
        max_iter = int(input("Enter maximum iterations (e.g., 100): ") or "100")
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError: Invalid input - {e}")
        return
    
    # Solve using Newton-Raphson
    try:
        root, iterations, converged = newton_raphson(
            f_expr, x0, tolerance, max_iter
        )
        
        # Additional analysis
        if converged:
            print("\nVerification:")
            print("-" * 80)
            x = sp.Symbol('x')
            f = parse_expr(f_expr, transformations='all')
            f_func = sp.lambdify(x, f, 'numpy')
            f_at_root = float(f_func(root))
            print(f"f({root:.10f}) = {f_at_root:.10e}")
            
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
