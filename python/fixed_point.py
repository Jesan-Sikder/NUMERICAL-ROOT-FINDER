"""
Fixed Point Iteration Method for Root Finding
==============================================

This module implements the Fixed Point Iteration algorithm to find roots of
nonlinear equations. The method transforms f(x) = 0 into the form x = g(x)
and iteratively computes x_{n+1} = g(x_n) until convergence.

Author: Numerical Methods Project
"""

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import Tuple, List, Dict


def fixed_point_iteration(
    g_expr: str,
    x0: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[float, List[Dict], bool]:
    """
    Perform Fixed Point Iteration to find the root of an equation.
    
    The method solves x = g(x) by iterating:
        x_{n+1} = g(x_n)
    
    Parameters:
    -----------
    g_expr : str
        The function g(x) as a string expression (e.g., 'cos(x)' for x = cos(x))
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
        - List of iteration data (iteration number, x value, error)
        - Boolean indicating if method converged
    
    Example:
    --------
    >>> root, iterations, converged = fixed_point_iteration('cos(x)', 0.5, 1e-6, 100)
    >>> print(f"Root: {root:.6f}")
    """
    
    # Define symbolic variable
    x = sp.Symbol('x')
    
    # Parse the function g(x)
    try:
        g = parse_expr(g_expr, transformations='all')
    except Exception as e:
        raise ValueError(f"Error parsing function: {e}")
    
    # Convert to numerical function
    g_func = sp.lambdify(x, g, 'numpy')
    
    # Initialize iteration tracking
    iterations_data = []
    x_current = x0
    converged = False
    
    print("\n" + "="*70)
    print("FIXED POINT ITERATION METHOD")
    print("="*70)
    print(f"Function g(x): {g}")
    print(f"Initial guess x0: {x0}")
    print(f"Tolerance: {tolerance}")
    print(f"Maximum iterations: {max_iterations}")
    print("="*70)
    print(f"{'Iteration':<12} {'x':<20} {'Error':<20}")
    print("-"*70)
    
    # Iteration 0 (initial guess)
    iterations_data.append({
        'iteration': 0,
        'x': x_current,
        'error': None
    })
    print(f"{0:<12} {x_current:<20.10f} {'N/A':<20}")
    
    # Perform iterations
    for iteration in range(1, max_iterations + 1):
        # Compute next value: x_{n+1} = g(x_n)
        try:
            x_next = float(g_func(x_current))
        except Exception as e:
            print(f"\nError evaluating function at x = {x_current}: {e}")
            break
        
        # Calculate error (absolute difference)
        error = abs(x_next - x_current)
        
        # Store iteration data
        iterations_data.append({
            'iteration': iteration,
            'x': x_next,
            'error': error
        })
        
        # Print iteration details
        print(f"{iteration:<12} {x_next:<20.10f} {error:<20.10e}")
        
        # Check convergence
        if error < tolerance:
            converged = True
            break
        
        # Update for next iteration
        x_current = x_next
    
    print("-"*70)
    
    if converged:
        print(f"\n✓ Converged after {iteration} iterations")
        print(f"✓ Root: x ≈ {x_next:.10f}")
    else:
        print(f"\n✗ Failed to converge within {max_iterations} iterations")
        print(f"✗ Last approximation: x ≈ {x_current:.10f}")
    
    print("="*70)
    
    final_root = x_next if converged else x_current
    return final_root, iterations_data, converged


def check_convergence_condition(g_expr: str, root_approx: float) -> Tuple[bool, float]:
    """
    Check the convergence condition for Fixed Point Iteration.
    
    The method converges if |g'(x)| < 1 at the root.
    
    Parameters:
    -----------
    g_expr : str
        The function g(x) as a string
    root_approx : float
        Approximation of the root
    
    Returns:
    --------
    Tuple[bool, float]
        - Boolean indicating if convergence condition is satisfied
        - Value of |g'(x)| at the root
    """
    x = sp.Symbol('x')
    g = parse_expr(g_expr, transformations='all')
    g_prime = sp.diff(g, x)
    g_prime_func = sp.lambdify(x, g_prime, 'numpy')
    
    derivative_value = abs(float(g_prime_func(root_approx)))
    is_convergent = derivative_value < 1
    
    return is_convergent, derivative_value


def main():
    """
    Main function demonstrating the Fixed Point Iteration method.
    Allows user input for interactive solving.
    """
    print("\n" + "="*70)
    print("FIXED POINT ITERATION - INTERACTIVE SOLVER")
    print("="*70)
    print("\nThis program finds the fixed point of x = g(x)")
    print("Example: To solve x = cos(x), enter g(x) = cos(x)")
    print()
    
    # Get user input
    try:
        g_expr = input("Enter g(x) (e.g., 'cos(x)', 'exp(-x)', '(x+3)**(1/2)'): ").strip()
        x0 = float(input("Enter initial guess x0: "))
        tolerance = float(input("Enter tolerance (e.g., 1e-6): ") or "1e-6")
        max_iter = int(input("Enter maximum iterations (e.g., 100): ") or "100")
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError: Invalid input - {e}")
        return
    
    # Solve using Fixed Point Iteration
    try:
        root, iterations, converged = fixed_point_iteration(
            g_expr, x0, tolerance, max_iter
        )
        
        # Check convergence condition
        print("\nConvergence Analysis:")
        print("-" * 70)
        is_convergent, derivative = check_convergence_condition(g_expr, root)
        print(f"|g'(x)| at root ≈ {derivative:.6f}")
        if is_convergent:
            print("✓ Convergence condition satisfied: |g'(x)| < 1")
        else:
            print("✗ Warning: |g'(x)| >= 1, method may not converge from all initial guesses")
        
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
