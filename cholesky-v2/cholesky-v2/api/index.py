"""
Cholesky Decomposition — PIT Numerical Methods Project
Single-file Vercel deployment: Flask app + inlined HTML
"""
import math
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
#  ALGORITHM
# ─────────────────────────────────────────────────────────────

def cholesky_decompose(matrix):
    n = len(matrix)
    A = [row[:] for row in matrix]

    for i, row in enumerate(A):
        if len(row) != n:
            raise ValueError("Matrix must be square.")
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > 1e-9:
                raise ValueError(
                    f"Matrix is not symmetric: "
                    f"A[{i+1}][{j+1}]={A[i][j]} ≠ A[{j+1}][{i+1}]={A[j][i]}"
                )

    L = [[0.0] * n for _ in range(n)]
    steps = []

    for j in range(n):
        sum_sq = sum(L[j][k] ** 2 for k in range(j))
        val = A[j][j] - sum_sq
        if val <= 1e-14:
            raise ValueError(
                f"Matrix is not positive-definite. "
                f"At column {j+1}: inner value = {val:.6g} ≤ 0"
            )
        L[j][j] = math.sqrt(val)
        steps.append({
            "type": "diagonal", "row": j+1, "col": j+1,
            "formula": (
                f"L[{j+1}][{j+1}] = sqrt({A[j][j]}"
                + (f" - {round(sum_sq,6)}" if j > 0 else "")
                + f") = sqrt({round(val,6)})"
            ),
            "sum_sq": round(sum_sq, 8),
            "val": round(val, 8),
            "result": round(L[j][j], 8),
        })
        for i in range(j + 1, n):
            sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
            numerator = A[i][j] - sum_prod
            L[i][j] = numerator / L[j][j]
            steps.append({
                "type": "off_diagonal", "row": i+1, "col": j+1,
                "formula": (
                    f"L[{i+1}][{j+1}] = ({A[i][j]}"
                    + (f" - {round(sum_prod,6)}" if j > 0 else "")
                    + f") / {round(L[j][j],6)}"
                ),
                "sum_prod": round(sum_prod, 8),
                "numerator": round(numerator, 8),
                "divisor": round(L[j][j], 8),
                "result": round(L[i][j], 8),
            })

    reconstructed = [
        [round(sum(L[r][k] * L[c][k] for k in range(n)), 6) for c in range(n)]
        for r in range(n)
    ]
    L_rounded = [[round(L[i][j], 6) for j in range(n)] for i in range(n)]
    return {"L": L_rounded, "steps": steps, "reconstructed": reconstructed, "n": n}


def solve_cholesky(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][k] * y[k] for k in range(i))
        y[i] = (b[i] - s) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / L[i][i]
    return [round(v, 6) for v in x], [round(v, 6) for v in y]


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json(force=True)
        raw   = data.get("matrix", [])
        solve = data.get("solve", False)
        b_vec = data.get("b", [])

        n = len(raw)
        if n < 2 or n > 6:
            return jsonify({"error": "Matrix size must be 2×2 to 6×6."}), 400

        matrix = []
        for i, row in enumerate(raw):
            if len(row) != n:
                return jsonify({"error": "Matrix must be square."}), 400
            parsed = []
            for j, val in enumerate(row):
                try:
                    parsed.append(float(val))
                except (TypeError, ValueError):
                    return jsonify({"error": f"Invalid value at row {i+1}, col {j+1}."}), 400
            matrix.append(parsed)

        result = cholesky_decompose(matrix)

        if solve and b_vec:
            if len(b_vec) != n:
                return jsonify({"error": "Vector b must match matrix size."}), 400
            try:
                b = [float(v) for v in b_vec]
            except (TypeError, ValueError):
                return jsonify({"error": "Vector b contains invalid values."}), 400
            x, y = solve_cholesky(result["L"], b)
            result["x"] = x
            result["y"] = y
            result["b"] = [round(v, 6) for v in b]

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def index(path):
    return HTML_PAGE, 200, {"Content-Type": "text/html"}


# ─────────────────────────────────────────────────────────────
#  INLINED HTML  (no templates folder needed on Vercel)
# ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Cholesky Decomposition | Numerical Methods</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<script>
window.MathJax={tex:{inlineMath:[['$','$'],['\\(','\\)']],displayMath:[['$$','$$'],['\\[','\\]']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre']}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
:root{
  --bg:#0d1117;--surface:#161b22;--surface2:#21262d;--border:#30363d;
  --accent:#e6a817;--accent2:#58a6ff;--text:#e6edf3;--muted:#8b949e;
  --success:#3fb950;--error:#f85149;
  --mono:'IBM Plex Mono',monospace;
  --serif:'DM Serif Display',Georgia,serif;
  --sans:'DM Sans',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.7;font-size:16px}

header{border-bottom:1px solid var(--border);background:linear-gradient(135deg,#0d1117 60%,#161022);padding:0 2rem;position:sticky;top:0;z-index:100}
.header-inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:1.2rem;padding:0.9rem 0}
.badge{background:var(--accent);color:#000;font-family:var(--mono);font-size:0.65rem;font-weight:700;padding:0.2rem 0.55rem;border-radius:4px;letter-spacing:0.08em;text-transform:uppercase}
header h1{font-family:var(--serif);font-size:1.25rem;font-weight:400}
nav{margin-left:auto;display:flex;gap:1.5rem}
nav a{color:var(--muted);text-decoration:none;font-size:0.85rem;font-weight:500;transition:color .2s}
nav a:hover{color:var(--accent)}

main{max-width:1100px;margin:0 auto;padding:3rem 2rem 6rem}
section{margin-bottom:4rem;animation:fadeUp .5s ease both}
section:nth-child(1){animation-delay:.05s}
section:nth-child(2){animation-delay:.15s}
section:nth-child(3){animation-delay:.25s}
section>p{color:#c9d1d9;max-width:780px;margin-bottom:1.2rem}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}

.section-label{display:flex;align-items:center;gap:.75rem;margin-bottom:1.5rem}
.section-label .num{font-family:var(--mono);font-size:.7rem;color:var(--accent);background:rgba(230,168,23,.1);border:1px solid rgba(230,168,23,.25);padding:.15rem .45rem;border-radius:4px}
.section-label h2{font-family:var(--serif);font-size:1.6rem;font-weight:400}

.math-block{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:6px;padding:1.25rem 1.5rem;margin:1.2rem 0;overflow-x:auto}
.math-block.center{text-align:center}

.algo-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.5rem;margin:1.5rem 0}
.algo-box h4{font-family:var(--mono);font-size:.78rem;color:var(--accent);letter-spacing:.08em;text-transform:uppercase;margin-bottom:.8rem}
.algo-step{display:flex;gap:1rem;padding:.5rem 0;border-bottom:1px solid var(--border);align-items:flex-start}
.algo-step:last-child{border-bottom:none}
.algo-step .sn{font-family:var(--mono);font-size:.7rem;color:var(--accent2);min-width:2rem;padding-top:.1rem}
.algo-step p{font-size:.9rem;color:#c9d1d9}

.props-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:1rem;margin:1.5rem 0}
.prop-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem}
.prop-card h5{font-family:var(--mono);font-size:.7rem;color:var(--accent2);letter-spacing:.08em;text-transform:uppercase;margin-bottom:.4rem}
.prop-card p{font-size:.88rem;color:#c9d1d9}

.example-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:2rem;margin-bottom:2rem}
.example-header{display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid var(--border)}
.ex-num{background:var(--accent);color:#000;font-family:var(--mono);font-weight:700;font-size:.75rem;width:2rem;height:2rem;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.example-header h3{font-family:var(--serif);font-size:1.2rem;font-weight:400}
.step-row{display:flex;gap:1rem;margin:.8rem 0;align-items:flex-start}
.sr-label{font-family:var(--mono);font-size:.7rem;background:rgba(88,166,255,.1);color:var(--accent2);border:1px solid rgba(88,166,255,.2);padding:.15rem .4rem;border-radius:4px;white-space:nowrap;margin-top:.35rem}
.sr-content{flex:1}
.sr-content p{font-size:.9rem;color:#c9d1d9;margin-bottom:.3rem}
.result-box{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.25);border-radius:6px;padding:1rem 1.25rem;margin-top:1rem}
.result-box h5{font-family:var(--mono);font-size:.7rem;color:var(--success);letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem}

.calc-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:2rem}
.calc-controls{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;margin-bottom:1.5rem}
.calc-controls label{font-size:.9rem;color:var(--muted);font-weight:500}
select,input[type=number]{background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:var(--mono);font-size:.88rem;padding:.45rem .8rem;outline:none;transition:border-color .2s}
select:focus,input:focus{border-color:var(--accent)}
.matrix-input-grid{display:inline-grid;gap:.4rem;margin:1rem 0}
.matrix-input-row{display:flex;gap:.4rem}
.matrix-input-row input{width:5.5rem;text-align:center}
.b-vec-wrap{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-top:1rem}
.b-vec-wrap label{font-size:.88rem;color:var(--muted)}
.b-vec-wrap input{width:5rem}
.toggle-row{display:flex;align-items:center;gap:.6rem;margin:1rem 0}
.toggle-row label{font-size:.88rem;color:var(--muted);cursor:pointer}
input[type=checkbox]{accent-color:var(--accent);width:1rem;height:1rem;cursor:pointer}
.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.55rem 1.3rem;border:none;border-radius:6px;font-family:var(--sans);font-weight:600;font-size:.88rem;cursor:pointer;transition:all .2s}
.btn-primary{background:var(--accent);color:#000}
.btn-primary:hover{background:#f0b82a;transform:translateY(-1px);box-shadow:0 4px 14px rgba(230,168,23,.3)}
.btn-secondary{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-secondary:hover{border-color:var(--muted);color:var(--text)}

#output{margin-top:2rem}
.out-sec{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:1.5rem;margin-bottom:1rem}
.out-sec h4{font-family:var(--mono);font-size:.72rem;color:var(--accent);letter-spacing:.1em;text-transform:uppercase;margin-bottom:1rem}
.mat-display{display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.mat-label{font-family:var(--mono);font-size:.9rem;color:var(--accent);font-weight:600}
.mat-grid{display:inline-block;border-left:2.5px solid var(--muted);border-right:2.5px solid var(--muted);padding:.2rem .6rem}
.mat-row{display:flex}
.mat-cell{font-family:var(--mono);font-size:.85rem;min-width:4.5rem;text-align:right;padding:.2rem .4rem}
.mat-cell.zero{color:var(--muted)}
.steps-tbl{width:100%;border-collapse:collapse;font-size:.8rem}
.steps-tbl th{text-align:left;padding:.5rem .7rem;font-family:var(--mono);font-size:.65rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid var(--border)}
.steps-tbl td{padding:.4rem .7rem;border-bottom:1px solid rgba(48,54,61,.5);font-family:var(--mono);color:#c9d1d9}
.steps-tbl tr:last-child td{border-bottom:none}
.steps-tbl tr.diag td:first-child{color:var(--accent)}
.steps-tbl tr.off td:first-child{color:var(--accent2)}
.chips{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem}
.chip{border-radius:6px;padding:.4rem .9rem;font-family:var(--mono);font-size:.85rem}
.chip-y{background:var(--surface);border:1px solid var(--border)}
.chip-x{background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.3);color:var(--success)}
.error-msg{background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.3);color:var(--error);border-radius:6px;padding:.8rem 1rem;font-size:.88rem;font-family:var(--mono)}
footer{border-top:1px solid var(--border);text-align:center;padding:1.5rem;color:var(--muted);font-size:.8rem}
@media(max-width:700px){main{padding:2rem 1rem 4rem}nav{display:none}.matrix-input-row input{width:4rem}}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <span class="badge">Topic #09</span>
    <h1>Cholesky Decomposition</h1>
    <nav>
      <a href="#discussion">Discussion</a>
      <a href="#examples">Examples</a>
      <a href="#calculator">Calculator</a>
    </nav>
  </div>
</header>

<main>

<!-- ── 1. DISCUSSION ─────────────────────────────────────── -->
<section id="discussion">
  <div class="section-label"><span class="num">01</span><h2>Mathematical Discussion</h2></div>
  <p>Cholesky decomposition factorises a <strong>symmetric positive-definite (SPD)</strong> matrix $A$ into a lower-triangular matrix $L$ and its transpose:</p>
  <div class="math-block center">$$A = L\,L^T$$</div>
  <p>It is roughly <strong>twice as fast</strong> as LU decomposition for SPD matrices and is unconditionally numerically stable — no pivoting is needed.</p>

  <div class="algo-box">
    <h4>Required Conditions</h4>
    <div class="algo-step"><span class="sn">C1</span><p><strong>Square:</strong> $A$ is $n\times n$.</p></div>
    <div class="algo-step"><span class="sn">C2</span><p><strong>Symmetric:</strong> $a_{ij}=a_{ji}$ for all $i,j$.</p></div>
    <div class="algo-step"><span class="sn">C3</span><p><strong>Positive-definite:</strong> $\mathbf{x}^TA\mathbf{x}>0$ for all non-zero $\mathbf{x}\in\mathbb{R}^n$.</p></div>
  </div>

  <div class="section-label" style="margin-top:2.5rem"><span class="num">—</span><h2 style="font-size:1.2rem">Derivation of Formulas</h2></div>
  <p>Expanding $A=LL^T$ entry by entry and solving column by column:</p>
  <div class="math-block"><strong>Diagonal entry</strong> $(i=j)$:
    $$l_{jj}=\sqrt{a_{jj}-\sum_{k=1}^{j-1}l_{jk}^2}$$
  </div>
  <div class="math-block"><strong>Sub-diagonal entry</strong> $(i>j)$:
    $$l_{ij}=\frac{1}{l_{jj}}\!\left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}\,l_{jk}\right)$$
  </div>

  <div class="props-grid">
    <div class="prop-card"><h5>Complexity</h5><p>$O(n^3/3)$ — about half of Gaussian elimination on a general matrix.</p></div>
    <div class="prop-card"><h5>Uniqueness</h5><p>Unique when all diagonal entries of $L$ are required to be positive.</p></div>
    <div class="prop-card"><h5>Stability</h5><p>Unconditionally stable for SPD matrices. No pivoting required.</p></div>
    <div class="prop-card"><h5>Applications</h5><p>Solving $Ax=b$, Monte Carlo sampling, least-squares, finite-element methods.</p></div>
  </div>

  <div class="section-label" style="margin-top:2rem"><span class="num">—</span><h2 style="font-size:1.2rem">Solving $Ax=b$ via Cholesky</h2></div>
  <div class="algo-box">
    <h4>Two-Phase Substitution</h4>
    <div class="algo-step"><span class="sn">01</span>
      <p><strong>Forward substitution</strong> — solve $L\mathbf{y}=\mathbf{b}$:
        $$y_i=\frac{1}{l_{ii}}\!\left(b_i-\sum_{k=1}^{i-1}l_{ik}\,y_k\right)$$
      </p>
    </div>
    <div class="algo-step"><span class="sn">02</span>
      <p><strong>Backward substitution</strong> — solve $L^T\mathbf{x}=\mathbf{y}$:
        $$x_i=\frac{1}{l_{ii}}\!\left(y_i-\sum_{k=i+1}^{n}l_{ki}\,x_k\right)$$
      </p>
    </div>
  </div>
</section>

<!-- ── 2. EXAMPLES ───────────────────────────────────────── -->
<section id="examples">
  <div class="section-label"><span class="num">02</span><h2>Worked Examples</h2></div>

  <!-- Example 1 -->
  <div class="example-card">
    <div class="example-header">
      <span class="ex-num">1</span>
      <div>
        <h3>Decompose a 3×3 SPD Matrix</h3>
        <p style="color:var(--muted);font-size:.85rem">Compute $L$ such that $A=LL^T$, then verify.</p>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Given</span>
      <div class="sr-content">
        <div class="math-block center">$$A=\begin{bmatrix}4&2&-2\\2&10&2\\-2&2&5\end{bmatrix}$$</div>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Step 1</span>
      <div class="sr-content">
        <p><strong>Column $j=1$</strong></p>
        <div class="math-block">
          $$l_{11}=\sqrt{4}=2$$
          $$l_{21}=\frac{2}{2}=1\qquad l_{31}=\frac{-2}{2}=-1$$
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Step 2</span>
      <div class="sr-content">
        <p><strong>Column $j=2$</strong></p>
        <div class="math-block">
          $$l_{22}=\sqrt{10-1^2}=\sqrt{9}=3$$
          $$l_{32}=\frac{2-(-1)(1)}{3}=\frac{3}{3}=1$$
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Step 3</span>
      <div class="sr-content">
        <p><strong>Column $j=3$</strong></p>
        <div class="math-block">$$l_{33}=\sqrt{5-(-1)^2-1^2}=\sqrt{3}\approx1.7321$$</div>
      </div>
    </div>
    <div class="result-box">
      <h5>Result</h5>
      <div class="math-block center" style="background:transparent;border:none;padding:0">
        $$L=\begin{bmatrix}2&0&0\\1&3&0\\-1&1&\sqrt{3}\end{bmatrix}$$
      </div>
      <p style="margin-top:.8rem;font-size:.88rem;color:#c9d1d9"><strong>Verification:</strong></p>
      <div class="math-block center" style="background:transparent;border:none">
        $$LL^T=\begin{bmatrix}2&0&0\\1&3&0\\-1&1&\sqrt3\end{bmatrix}\begin{bmatrix}2&1&-1\\0&3&1\\0&0&\sqrt3\end{bmatrix}=\begin{bmatrix}4&2&-2\\2&10&2\\-2&2&5\end{bmatrix}=A\checkmark$$
      </div>
    </div>
  </div>

  <!-- Example 2 -->
  <div class="example-card">
    <div class="example-header">
      <span class="ex-num">2</span>
      <div>
        <h3>Solve $Ax=b$ Using Cholesky</h3>
        <p style="color:var(--muted);font-size:.85rem">Full factorisation then forward/backward substitution.</p>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Given</span>
      <div class="sr-content">
        <div class="math-block center">
          $$A=\begin{bmatrix}25&15&-5\\15&18&0\\-5&0&11\end{bmatrix},\quad\mathbf{b}=\begin{bmatrix}35\\33\\6\end{bmatrix}$$
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Step 1</span>
      <div class="sr-content">
        <p><strong>Cholesky Factorisation</strong></p>
        <div class="math-block">
          $$l_{11}=\sqrt{25}=5$$
          $$l_{21}=\tfrac{15}{5}=3,\quad l_{31}=\tfrac{-5}{5}=-1$$
          $$l_{22}=\sqrt{18-9}=3$$
          $$l_{32}=\tfrac{0-(-1)(3)}{3}=1$$
          $$l_{33}=\sqrt{11-1-1}=3$$
        </div>
        <div class="math-block center">$$L=\begin{bmatrix}5&0&0\\3&3&0\\-1&1&3\end{bmatrix}$$</div>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Step 2</span>
      <div class="sr-content">
        <p><strong>Forward substitution $L\mathbf{y}=\mathbf{b}$</strong></p>
        <div class="math-block">
          $$y_1=\tfrac{35}{5}=7$$
          $$y_2=\tfrac{33-3(7)}{3}=4$$
          $$y_3=\tfrac{6-(-1)(7)-(1)(4)}{3}=3$$
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="sr-label">Step 3</span>
      <div class="sr-content">
        <p><strong>Backward substitution $L^T\mathbf{x}=\mathbf{y}$</strong></p>
        <div class="math-block">
          $$x_3=\tfrac{3}{3}=1$$
          $$x_2=\tfrac{4-(1)(1)}{3}=1$$
          $$x_1=\tfrac{7-(3)(1)-(-1)(1)}{5}=1$$
        </div>
      </div>
    </div>
    <div class="result-box">
      <h5>Solution</h5>
      <div class="math-block center" style="background:transparent;border:none;padding:0">$$\mathbf{x}=[1,1,1]^T$$</div>
      <p style="margin-top:.8rem;font-size:.88rem;color:#c9d1d9">
        <strong>Check:</strong> $25+15-5=35\ \checkmark\quad 15+18+0=33\ \checkmark\quad -5+0+11=6\ \checkmark$
      </p>
    </div>
  </div>
</section>

<!-- ── 3. CALCULATOR ──────────────────────────────────────── -->
<section id="calculator">
  <div class="section-label"><span class="num">03</span><h2>Interactive Calculator</h2></div>
  <p>Enter a symmetric positive-definite matrix. Values mirror automatically across the diagonal. Optionally solve $A\mathbf{x}=\mathbf{b}$.</p>

  <div class="calc-box">
    <div class="calc-controls">
      <label for="size">Matrix size:</label>
      <select id="size" onchange="buildMatrix()">
        <option value="2">2 × 2</option>
        <option value="3" selected>3 × 3</option>
        <option value="4">4 × 4</option>
        <option value="5">5 × 5</option>
        <option value="6">6 × 6</option>
      </select>
      <button class="btn btn-secondary" onclick="fillEx1()">Load Example 1</button>
      <button class="btn btn-secondary" onclick="fillEx2()">Load Example 2</button>
      <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
    </div>

    <label style="font-size:.85rem;color:var(--muted)">Matrix $A$ — entering a value mirrors it across the diagonal automatically.</label>
    <div id="matrix-input" class="matrix-input-grid"></div>

    <div class="toggle-row">
      <input type="checkbox" id="solve-toggle" onchange="toggleB()"/>
      <label for="solve-toggle">Also solve $A\mathbf{x}=\mathbf{b}$</label>
    </div>
    <div id="b-section" style="display:none">
      <div class="b-vec-wrap">
        <label>Vector $\mathbf{b}$:</label>
        <div id="b-inputs" style="display:flex;gap:.4rem;flex-wrap:wrap"></div>
      </div>
    </div>

    <div style="margin-top:1.25rem;display:flex;gap:.75rem;flex-wrap:wrap">
      <button class="btn btn-primary" onclick="runCalc()">&#9654; Decompose</button>
    </div>
    <div id="output"></div>
  </div>
</section>

</main>
<footer>Cholesky Decomposition &middot; PIT Project &ndash; Numerical Methods &middot; Flask + MathJax</footer>

<script>
let N = 3;

function buildMatrix() {
  N = parseInt(document.getElementById('size').value);
  const grid = document.getElementById('matrix-input');
  grid.innerHTML = '';
  for (let i = 0; i < N; i++) {
    const row = document.createElement('div');
    row.className = 'matrix-input-row';
    for (let j = 0; j < N; j++) {
      const inp = document.createElement('input');
      inp.type = 'number'; inp.step = 'any';
      inp.id = `m_${i}_${j}`;
      inp.placeholder = `a${i+1}${j+1}`;
      inp.addEventListener('input', () => {
        if (i !== j) {
          const m = document.getElementById(`m_${j}_${i}`);
          if (m) m.value = inp.value;
        }
      });
      row.appendChild(inp);
    }
    grid.appendChild(row);
  }
  buildBVec();
}

function buildBVec() {
  const c = document.getElementById('b-inputs');
  c.innerHTML = '';
  for (let i = 0; i < N; i++) {
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = 'any';
    inp.id = `b_${i}`; inp.placeholder = `b${i+1}`;
    c.appendChild(inp);
  }
}

function toggleB() {
  document.getElementById('b-section').style.display =
    document.getElementById('solve-toggle').checked ? 'block' : 'none';
}

function setMatrix(data) {
  document.getElementById('size').value = data.length;
  buildMatrix();
  for (let i = 0; i < data.length; i++)
    for (let j = 0; j < data.length; j++)
      document.getElementById(`m_${i}_${j}`).value = data[i][j];
}

function fillEx1() {
  setMatrix([[4,2,-2],[2,10,2],[-2,2,5]]);
  document.getElementById('solve-toggle').checked = false; toggleB();
}
function fillEx2() {
  setMatrix([[25,15,-5],[15,18,0],[-5,0,11]]);
  document.getElementById('solve-toggle').checked = true; toggleB();
  [35,33,6].forEach((v,i) => { document.getElementById(`b_${i}`).value = v; });
}
function clearAll() { buildMatrix(); document.getElementById('output').innerHTML = ''; }

function readMatrix() {
  const mat = [];
  for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < N; j++) {
      const v = parseFloat(document.getElementById(`m_${i}_${j}`).value);
      if (isNaN(v)) throw new Error(`Cell (${i+1},${j+1}) is empty or invalid.`);
      row.push(v);
    }
    mat.push(row);
  }
  return mat;
}
function readB() {
  const b = [];
  for (let i = 0; i < N; i++) {
    const v = parseFloat(document.getElementById(`b_${i}`).value);
    if (isNaN(v)) throw new Error(`b[${i+1}] is empty or invalid.`);
    b.push(v);
  }
  return b;
}

async function runCalc() {
  const out = document.getElementById('output');
  out.innerHTML = '<p style="color:var(--muted);padding:1rem 0;font-size:.88rem">Computing...</p>';
  try {
    const matrix = readMatrix();
    const solve  = document.getElementById('solve-toggle').checked;
    const payload = { matrix, solve };
    if (solve) payload.b = readB();

    const resp = await fetch('/calculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    // Check content type before parsing
    const ct = resp.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
      const txt = await resp.text();
      throw new Error('Server returned non-JSON response. Check Vercel logs.\n\n' + txt.slice(0,200));
    }

    const data = await resp.json();
    if (data.error) { out.innerHTML = `<div class="error-msg">&#9888; ${data.error}</div>`; return; }
    renderOutput(data);
  } catch(e) {
    out.innerHTML = `<div class="error-msg">&#9888; ${e.message}</div>`;
  }
}

function matHTML(M, label) {
  const n = M.length;
  let s = `<div class="mat-display"><span class="mat-label">${label} =</span><div class="mat-grid">`;
  for (let i = 0; i < n; i++) {
    s += '<div class="mat-row">';
    for (let j = 0; j < n; j++) {
      const v = M[i][j];
      s += `<span class="mat-cell ${v===0?'zero':''}">${v}</span>`;
    }
    s += '</div>';
  }
  return s + '</div></div>';
}

function renderOutput(data) {
  const out = document.getElementById('output');
  let h = '';

  h += `<div class="out-sec"><h4>Lower Triangular Factor L</h4>${matHTML(data.L,'L')}</div>`;

  h += `<div class="out-sec"><h4>Step-by-Step Computation</h4>
    <div style="overflow-x:auto"><table class="steps-tbl">
    <thead><tr><th>Entry</th><th>Type</th><th>Formula</th><th>&#x3a3; Term</th><th>Result</th></tr></thead><tbody>`;
  for (const s of data.steps) {
    const e = `L[${s.row}][${s.col}]`;
    if (s.type === 'diagonal') {
      h += `<tr class="diag"><td>${e}</td><td>Diagonal</td><td>${s.formula}</td><td>&#x3a3;k&#xb2;=${s.sum_sq}</td><td><strong>${s.result}</strong></td></tr>`;
    } else {
      h += `<tr class="off"><td>${e}</td><td>Off-diag</td><td>${s.formula}</td><td>&#x3a3;prod=${s.sum_prod} &divide; ${s.divisor}</td><td><strong>${s.result}</strong></td></tr>`;
    }
  }
  h += '</tbody></table></div></div>';

  h += `<div class="out-sec"><h4>Verification &mdash; LL&#7488; = A</h4>${matHTML(data.reconstructed,'LL&#7488;')}</div>`;

  if (data.x) {
    h += `<div class="out-sec"><h4>System Solution A&#x78; = b</h4>
      <p style="font-size:.85rem;color:var(--muted);margin-bottom:.6rem">Intermediate y &nbsp;(from Ly = b):</p>
      <div class="chips">${data.y.map((v,i)=>`<span class="chip chip-y">y${i+1} = ${v}</span>`).join('')}</div>
      <p style="font-size:.85rem;color:var(--muted);margin:.8rem 0 .6rem">Solution x &nbsp;(from L&#7488;x = y):</p>
      <div class="chips">${data.x.map((v,i)=>`<span class="chip chip-x">x${i+1} = ${v}</span>`).join('')}</div>
    </div>`;
  }

  out.innerHTML = h;
  if (window.MathJax) MathJax.typesetPromise([out]);
}

buildMatrix();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
