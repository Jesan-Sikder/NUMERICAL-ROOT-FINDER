/**
 * Numerical Methods - Root Finding Algorithms
 * JavaScript Implementation for Browser-based Computation
 */

// Mathematical function parser and evaluator
class MathParser {
    /**
     * Parse and evaluate mathematical expressions
     */
    static parse(expr) {
        // Replace common mathematical notation with JavaScript equivalents
        expr = expr.replace(/\^/g, '**');  // Power operator
        expr = expr.replace(/sin/g, 'Math.sin');
        expr = expr.replace(/cos/g, 'Math.cos');
        expr = expr.replace(/tan/g, 'Math.tan');
        expr = expr.replace(/exp/g, 'Math.exp');
        expr = expr.replace(/log/g, 'Math.log');
        expr = expr.replace(/sqrt/g, 'Math.sqrt');
        expr = expr.replace(/abs/g, 'Math.abs');
        expr = expr.replace(/pi/g, 'Math.PI');
        expr = expr.replace(/\be\b/g, 'Math.E');  // Euler's number (standalone only)
        
        return expr;
    }

    /**
     * Create a function from expression string
     */
    static createFunction(exprStr) {
        const parsedExpr = this.parse(exprStr);
        return new Function('x', `return ${parsedExpr};`);
    }

    /**
     * Numerical derivative using central difference formula
     */
    static derivative(func, x, h = 1e-7) {
        return (func(x + h) - func(x - h)) / (2 * h);
    }
}

/**
 * Newton-Raphson Method Implementation
 */
class NewtonRaphson {
    constructor(funcExpr, x0, tolerance, maxIterations) {
        this.funcExpr = funcExpr;
        this.func = MathParser.createFunction(funcExpr);
        this.x0 = x0;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.iterations = [];
        this.converged = false;
        this.finalRoot = null;
    }

    solve() {
        let xCurrent = this.x0;
        let fCurrent = this.func(xCurrent);

        // Store initial iteration
        this.iterations.push({
            iteration: 0,
            x: xCurrent,
            fx: fCurrent,
            error: null
        });

        // Perform iterations
        for (let i = 1; i <= this.maxIterations; i++) {
            // Compute derivative numerically
            const fPrime = MathParser.derivative(this.func, xCurrent);

            // Check for zero derivative
            if (Math.abs(fPrime) < 1e-12) {
                throw new Error(`Derivative too close to zero at x = ${xCurrent}. Cannot continue.`);
            }

            // Newton-Raphson formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
            const xNext = xCurrent - (fCurrent / fPrime);
            const fNext = this.func(xNext);

            // Calculate error
            const error = Math.abs(xNext - xCurrent);

            // Store iteration data
            this.iterations.push({
                iteration: i,
                x: xNext,
                fx: fNext,
                error: error
            });

            // Check convergence
            if (error < this.tolerance || Math.abs(fNext) < this.tolerance) {
                this.converged = true;
                this.finalRoot = xNext;
                break;
            }

            // Update for next iteration
            xCurrent = xNext;
            fCurrent = fNext;
        }

        if (!this.converged) {
            this.finalRoot = xCurrent;
        }

        return {
            root: this.finalRoot,
            iterations: this.iterations,
            converged: this.converged
        };
    }
}

/**
 * Fixed Point Iteration Implementation
 */
class FixedPointIteration {
    constructor(gExpr, x0, tolerance, maxIterations) {
        this.gExpr = gExpr;
        this.gFunc = MathParser.createFunction(gExpr);
        this.x0 = x0;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.iterations = [];
        this.converged = false;
        this.finalRoot = null;
    }

    solve() {
        let xCurrent = this.x0;

        // Store initial iteration
        this.iterations.push({
            iteration: 0,
            x: xCurrent,
            error: null
        });

        // Perform iterations
        for (let i = 1; i <= this.maxIterations; i++) {
            // Fixed point iteration: x_{n+1} = g(x_n)
            const xNext = this.gFunc(xCurrent);

            // Calculate error
            const error = Math.abs(xNext - xCurrent);

            // Store iteration data
            this.iterations.push({
                iteration: i,
                x: xNext,
                error: error
            });

            // Check convergence
            if (error < this.tolerance) {
                this.converged = true;
                this.finalRoot = xNext;
                break;
            }

            // Update for next iteration
            xCurrent = xNext;
        }

        if (!this.converged) {
            this.finalRoot = xCurrent;
        }

        return {
            root: this.finalRoot,
            iterations: this.iterations,
            converged: this.converged
        };
    }
}

/**
 * UI Update Functions
 */

function updateMethodDescription() {
    const method = document.getElementById('method').value;
    const descriptionDiv = document.getElementById('method-description');
    const functionLabel = document.getElementById('function-label');

    if (method === 'newton') {
        functionLabel.textContent = 'Enter function f(x):';
        descriptionDiv.innerHTML = `
            <p><strong>Newton-Raphson Method</strong></p>
            <p>Finds roots of f(x) = 0 using the iteration formula:</p>
            <p style="text-align: center; font-style: italic; margin: 10px 0;">
                x<sub>n+1</sub> = x<sub>n</sub> - f(x<sub>n</sub>)/f'(x<sub>n</sub>)
            </p>
            <p>The derivative is computed numerically. Fast convergence (quadratic) near the root.</p>
        `;
    } else {
        functionLabel.textContent = 'Enter function g(x):';
        descriptionDiv.innerHTML = `
            <p><strong>Fixed Point Iteration</strong></p>
            <p>Finds fixed points where x = g(x) using simple iteration:</p>
            <p style="text-align: center; font-style: italic; margin: 10px 0;">
                x<sub>n+1</sub> = g(x<sub>n</sub>)
            </p>
            <p>To solve f(x) = 0, rearrange as x = g(x). Converges if |g'(x)| < 1 near the fixed point.</p>
        `;
    }
}

function loadExample(exampleNum) {
    const methodSelect = document.getElementById('method');
    const functionInput = document.getElementById('function-input');
    const initialGuess = document.getElementById('initial-guess');

    switch (exampleNum) {
        case 1:
            // x^2 - 4 = 0 (roots at x = ±2)
            methodSelect.value = 'newton';
            functionInput.value = 'x^2 - 4';
            initialGuess.value = '1';
            break;
        case 2:
            // cos(x) - x = 0
            methodSelect.value = 'newton';
            functionInput.value = 'cos(x) - x';
            initialGuess.value = '0.5';
            break;
        case 3:
            // x^3 - x - 2 = 0
            methodSelect.value = 'newton';
            functionInput.value = 'x^3 - x - 2';
            initialGuess.value = '1.5';
            break;
        case 4:
            // e^(-x) - x = 0, use fixed point: x = e^(-x)
            methodSelect.value = 'fixedpoint';
            functionInput.value = 'exp(-x)';
            initialGuess.value = '0.5';
            break;
    }
    updateMethodDescription();
}

function displayResults(result, method) {
    const resultsSection = document.getElementById('results-section');
    const summaryDiv = document.getElementById('summary');
    const tbody = document.getElementById('iteration-tbody');
    const col3Header = document.getElementById('col3-header');

    // Clear previous results
    tbody.innerHTML = '';

    // Update column header based on method
    if (method === 'newton') {
        col3Header.textContent = 'f(x)';
    } else {
        col3Header.textContent = 'g(x)';
    }

    // Display summary
    const statusClass = result.converged ? 'converged' : 'not-converged';
    const statusText = result.converged ? '✓ Converged' : '✗ Did not converge';
    
    summaryDiv.innerHTML = `
        <p><strong>Status:</strong> <span class="${statusClass}">${statusText}</span></p>
        <p><strong>Number of Iterations:</strong> ${result.iterations.length - 1}</p>
        <p><strong>Final Root:</strong> x ≈ ${result.root.toFixed(10)}</p>
        ${method === 'newton' ? `<p><strong>f(root):</strong> ${result.iterations[result.iterations.length - 1].fx.toExponential(6)}</p>` : ''}
    `;

    // Display iteration table
    result.iterations.forEach(iter => {
        const row = tbody.insertRow();
        
        const cell1 = row.insertCell(0);
        cell1.textContent = iter.iteration;

        const cell2 = row.insertCell(1);
        cell2.textContent = iter.x.toFixed(10);

        const cell3 = row.insertCell(2);
        if (method === 'newton') {
            cell3.textContent = iter.fx.toExponential(6);
        } else {
            cell3.textContent = iter.iteration === 0 ? 'N/A' : iter.x.toFixed(10);
        }

        const cell4 = row.insertCell(3);
        cell4.textContent = iter.error === null ? 'N/A' : iter.error.toExponential(6);
    });

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = '⚠ Error: ' + message;
    errorDiv.style.display = 'block';
    
    // Hide error after 5 seconds
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function hideError() {
    const errorDiv = document.getElementById('error-message');
    errorDiv.style.display = 'none';
}

/**
 * Main solve function
 */
function solve() {
    hideError();

    try {
        // Get input values
        const method = document.getElementById('method').value;
        const functionExpr = document.getElementById('function-input').value.trim();
        const x0 = parseFloat(document.getElementById('initial-guess').value);
        const tolerance = parseFloat(document.getElementById('tolerance').value);
        const maxIterations = parseInt(document.getElementById('max-iterations').value);

        // Validate inputs
        if (!functionExpr) {
            throw new Error('Please enter a function');
        }
        if (isNaN(x0)) {
            throw new Error('Initial guess must be a valid number');
        }
        if (isNaN(tolerance) || tolerance <= 0) {
            throw new Error('Tolerance must be a positive number');
        }
        if (isNaN(maxIterations) || maxIterations < 1) {
            throw new Error('Maximum iterations must be at least 1');
        }

        // Solve using selected method
        let result;
        if (method === 'newton') {
            const solver = new NewtonRaphson(functionExpr, x0, tolerance, maxIterations);
            result = solver.solve();
        } else {
            const solver = new FixedPointIteration(functionExpr, x0, tolerance, maxIterations);
            result = solver.solve();
        }

        // Display results
        displayResults(result, method);

    } catch (error) {
        showError(error.message);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    updateMethodDescription();
});
