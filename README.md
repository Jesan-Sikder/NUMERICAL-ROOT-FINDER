# Numerical Methods - Root Finding Algorithms

A comprehensive undergraduate project implementing two fundamental numerical methods for finding roots of nonlinear equations: **Fixed Point Iteration** and the **Newton-Raphson Method**.

## 📋 Project Overview

This project provides:
- **Python implementations** with SymPy for automatic differentiation
- **Interactive web application** running entirely in the browser
- **Complete documentation** with theory and deployment instructions
- **Educational examples** demonstrating both methods

## 🏗️ Project Structure

```
project/
│
├── python/
│   ├── newton_raphson.py      # Newton-Raphson implementation
│   └── fixed_point.py          # Fixed Point Iteration implementation
│
├── website/
│   ├── index.html              # Web interface
│   ├── script.js               # JavaScript implementations
│   └── style.css               # Styling
│
└── README.md                   # This file
```

## 📚 Theory

### Newton-Raphson Method

The Newton-Raphson method (also called Newton's method) is an iterative algorithm for finding successively better approximations to the roots of a real-valued function.

#### Algorithm

Given a function **f(x)** and an initial guess **x₀**, the method uses the iteration:

```
x_{n+1} = x_n - f(x_n) / f'(x_n)
```

Where:
- **x_n** is the current approximation
- **f(x_n)** is the function value at x_n
- **f'(x_n)** is the derivative at x_n
- **x_{n+1}** is the next approximation

#### Steps

1. **Start** with an initial guess x₀
2. **Compute** f(x_n) and f'(x_n)
3. **Calculate** the next approximation: x_{n+1} = x_n - f(x_n)/f'(x_n)
4. **Check convergence**: If |x_{n+1} - x_n| < ε (tolerance), stop
5. **Repeat** from step 2 if not converged

#### Convergence

**Advantages:**
- **Quadratic convergence** when close to the root
- Very fast for well-behaved functions
- Requires only one function evaluation per iteration (derivative can be computed analytically)

**Conditions for Convergence:**
- Initial guess x₀ must be sufficiently close to the actual root
- f'(x) ≠ 0 in the neighborhood of the root
- f''(x) must be continuous and bounded

**Limitations:**
- Requires computation of derivative
- May fail if derivative is zero or very small
- May diverge with poor initial guess
- Can cycle or oscillate for certain functions

#### Example

Find the root of **f(x) = x² - 4 = 0** starting from x₀ = 1:

```
Iteration 0: x₀ = 1.0000000000
Iteration 1: x₁ = 2.5000000000 (f'(x) = 2x)
Iteration 2: x₂ = 2.0500000000
Iteration 3: x₃ = 2.0006097561
Iteration 4: x₄ = 2.0000000929
```

Root: **x ≈ 2.0000** (converged in 4 iterations)

---

### Fixed Point Iteration

Fixed Point Iteration is a method to find solutions to equations of the form **x = g(x)**, called fixed points.

#### Algorithm

To solve **f(x) = 0**, rearrange it into the form **x = g(x)**, then iterate:

```
x_{n+1} = g(x_n)
```

Where:
- **x_n** is the current approximation
- **g(x_n)** is the function value
- **x_{n+1}** is the next approximation

#### Steps

1. **Rearrange** f(x) = 0 into x = g(x)
2. **Start** with an initial guess x₀
3. **Compute** x_{n+1} = g(x_n)
4. **Check convergence**: If |x_{n+1} - x_n| < ε (tolerance), stop
5. **Repeat** from step 3 if not converged

#### Convergence

**Advantages:**
- Simple to implement
- No derivative required
- Works for a wide class of functions

**Convergence Theorem:**

The iteration **x_{n+1} = g(x_n)** converges to a fixed point if:
1. **g(x)** is continuous in an interval [a, b] containing the fixed point
2. **|g'(x)| < 1** for all x in [a, b]

**Convergence Rate:**
- **Linear convergence**: Error decreases by a constant factor each iteration
- Slower than Newton-Raphson (which has quadratic convergence)

**Limitations:**
- Convergence depends heavily on the choice of g(x)
- May diverge if |g'(x)| ≥ 1
- Multiple rearrangements possible; some converge, others don't
- Generally slower than Newton-Raphson

#### Example

Solve **e^(-x) - x = 0** by rearranging to **x = e^(-x)** (so g(x) = e^(-x)):

```
Iteration 0: x₀ = 0.5000000000
Iteration 1: x₁ = 0.6065306597
Iteration 2: x₂ = 0.5452392118
Iteration 3: x₃ = 0.5796123355
Iteration 4: x₄ = 0.5600646279
Iteration 5: x₅ = 0.5711721155
...
Iteration 10: x₁₀ = 0.5671432904
```

Root: **x ≈ 0.5671** (converged in ~10-15 iterations)

---

### Comparison

| Aspect | Newton-Raphson | Fixed Point Iteration |
|--------|----------------|----------------------|
| **Convergence Rate** | Quadratic (very fast) | Linear (slower) |
| **Derivative Required** | Yes (f'(x) needed) | No |
| **Complexity** | Higher | Lower |
| **Convergence Condition** | Good initial guess | \|g'(x)\| < 1 |
| **Typical Iterations** | 3-5 iterations | 10-30 iterations |
| **Best Use Case** | When derivative is easy to compute | When derivative is complex or unknown |

---

## 💻 Python Implementation

### Requirements

```bash
pip install sympy numpy
```

### Newton-Raphson Method

```python
python python/newton_raphson.py
```

**Features:**
- Accepts any mathematical expression
- Automatic derivative computation using SymPy
- Detailed iteration table
- Convergence analysis

**Example Usage:**

```python
from newton_raphson import newton_raphson

# Find root of x^2 - 4 = 0
root, iterations, converged = newton_raphson(
    f_expr='x**2 - 4',
    x0=1.0,
    tolerance=1e-6,
    max_iterations=100
)

print(f"Root: {root}")
```

### Fixed Point Iteration

```python
python python/fixed_point.py
```

**Features:**
- Accepts any mathematical expression for g(x)
- Convergence condition checking
- Detailed iteration tracking
- Error analysis

**Example Usage:**

```python
from fixed_point import fixed_point_iteration

# Find fixed point of x = cos(x)
root, iterations, converged = fixed_point_iteration(
    g_expr='cos(x)',
    x0=0.5,
    tolerance=1e-6,
    max_iterations=100
)

print(f"Fixed point: {root}")
```

---

## 🌐 Web Application

The web application provides an interactive interface to visualize and compare both methods.

### Features

- ✅ **Interactive input** for function, initial guess, tolerance
- ✅ **Real-time computation** entirely in browser (no server required)
- ✅ **Iteration table** showing convergence
- ✅ **Quick examples** for common problems
- ✅ **Theory section** with mathematical background
- ✅ **Responsive design** for mobile and desktop
- ✅ **Beautiful UI** with gradient styling

### Supported Functions

The web application supports standard mathematical functions:

- Basic operators: `+`, `-`, `*`, `/`, `^` (power)
- Trigonometric: `sin`, `cos`, `tan`
- Exponential: `exp`, `log`
- Other: `sqrt`, `abs`, `pi`, `e`

### Example Problems

1. **Quadratic Equation**: `x^2 - 4` → roots at x = ±2
2. **Transcendental**: `cos(x) - x` → root at x ≈ 0.739
3. **Polynomial**: `x^3 - x - 2` → root at x ≈ 1.521
4. **Exponential**: `exp(-x) - x` → use Fixed Point with g(x) = exp(-x)

### Local Testing

Open `website/index.html` directly in a web browser, or use a local server:

```bash
# Using Python's built-in server
cd website
python -m http.server 8000

# Visit http://localhost:8000
```

---

## 🚀 Deployment on Netlify

### Method 1: Drag and Drop (Easiest)

1. **Create a Netlify account** at [netlify.com](https://www.netlify.com/)
2. **Drag and drop** the `website` folder to Netlify's deployment interface
3. **Done!** Your site is live at `https://random-name.netlify.app`

### Method 2: Git Integration (Recommended)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Add numerical methods project"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

2. **Connect to Netlify**:
   - Go to [app.netlify.com](https://app.netlify.com/)
   - Click "Add new site" → "Import an existing project"
   - Choose your Git provider (GitHub, GitLab, Bitbucket)
   - Select your repository

3. **Configure Build Settings**:
   - **Base directory**: `project/website`
   - **Build command**: (leave empty)
   - **Publish directory**: `project/website`

4. **Deploy**:
   - Click "Deploy site"
   - Netlify will build and deploy automatically
   - Your site will be live at `https://your-site-name.netlify.app`

### Method 3: Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Navigate to project
cd website

# Deploy
netlify deploy --prod

# Follow the prompts to link your site
```

### Custom Domain (Optional)

1. Go to your site's settings in Netlify
2. Click "Domain management"
3. Add your custom domain
4. Follow DNS configuration instructions

---

## 🎓 Educational Use

### Learning Objectives

Students will learn:
- Implementation of iterative numerical methods
- Convergence analysis and error estimation
- Trade-offs between different algorithms
- Practical application of calculus (derivatives)
- Web development with HTML/CSS/JavaScript
- Python programming with SymPy

### Suggested Exercises

1. **Modify convergence criteria**: Try different stopping conditions
2. **Add new methods**: Implement Secant method or Bisection
3. **Visualize convergence**: Add graphs showing iteration progress
4. **Compare performance**: Count function evaluations for each method
5. **Test edge cases**: Functions with multiple roots, no roots, etc.

### Extension Ideas

- Add graphical visualization of iterations
- Implement multiple root finding
- Add error bounds and accuracy estimation
- Compare computational complexity
- Handle complex-valued functions

---

## 📝 Code Quality

### Python Code
- **Clean and commented**: Every function has docstrings
- **Type hints**: Parameters and return types documented
- **Error handling**: Graceful handling of edge cases
- **Educational**: Print statements show iteration progress

### JavaScript Code
- **Modular design**: Separate classes for each method
- **Object-oriented**: Clean class structure
- **Error handling**: User-friendly error messages
- **Comments**: Explaining key algorithms

---

## 🧪 Testing

### Test Cases

**Newton-Raphson**:
```python
# Test 1: x^2 - 4 = 0
root = newton_raphson('x**2 - 4', 1.0)  # Expected: 2.0

# Test 2: x^3 - x - 2 = 0
root = newton_raphson('x**3 - x - 2', 1.5)  # Expected: ~1.521

# Test 3: cos(x) - x = 0
root = newton_raphson('cos(x) - x', 0.5)  # Expected: ~0.739
```

**Fixed Point Iteration**:
```python
# Test 1: x = cos(x)
root = fixed_point_iteration('cos(x)', 0.5)  # Expected: ~0.739

# Test 2: x = exp(-x)
root = fixed_point_iteration('exp(-x)', 0.5)  # Expected: ~0.567

# Test 3: x = (x + 3)^(1/2)
root = fixed_point_iteration('(x + 3)**(1/2)', 2.0)  # Expected: 3.0
```

---

## 📖 References

1. **Burden, R. L., & Faires, J. D.** (2010). *Numerical Analysis* (9th ed.). Brooks/Cole.
2. **Chapra, S. C., & Canale, R. P.** (2015). *Numerical Methods for Engineers* (7th ed.). McGraw-Hill.
3. **Press, W. H., et al.** (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.

---

## 🤝 Contributing

This is an educational project. Suggestions for improvement are welcome:
- Additional numerical methods
- Better visualizations
- More test cases
- Documentation improvements

---

## 📄 License

This project is provided for educational purposes. Feel free to use and modify for learning.

---

## 👨‍🎓 Author

**Undergraduate Numerical Methods Project**
MD Habibur Rahman Jesan



---


## 🔗 Links

- **Live Demo**: [LINK](https://jesan.netlify.app)
- **GitHub**: [LINK](https://github.com/Jesan-Sikder/NUMERICAL-ROOT-FINDER)
- **Documentation**: See theory sections above

---

## 💡 Tips for Students

1. **Start simple**: Test with functions you can verify by hand (like x² - 4)
2. **Visualize**: Sketch the function to understand where roots might be
3. **Try different guesses**: See how initial guess affects convergence
4. **Compare methods**: Use both methods on the same problem
5. **Understand failure**: Learn from cases where methods don't converge

---

## ❓ FAQ

**Q: Why doesn't my function converge?**
- Check your initial guess (try different values)
- For Fixed Point: verify |g'(x)| < 1 condition
- For Newton-Raphson: ensure f'(x) ≠ 0 near root

**Q: How do I rearrange f(x) = 0 to x = g(x)?**
- Example: x² - x - 2 = 0 → x = x² - 2 (g(x) = x² - 2)
- Or: x = (x + 2)^(1/2) (another valid rearrangement)
- Different rearrangements may have different convergence properties

**Q: Which method should I use?**
- Use Newton-Raphson for faster convergence when derivative is easy
- Use Fixed Point when derivative is complex or you prefer simplicity

---

**Happy Computing! 🎉**
