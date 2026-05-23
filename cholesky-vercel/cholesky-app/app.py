"""
Cholesky Decomposition - Flask Web Application
PIT Project – Numerical Methods Online Calculator
Vercel-compatible entry point (variable must be named `app`)
"""

import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# ── Core Algorithm ────────────────────────────────────────────────────────────

def cholesky_decompose(matrix):
    """
    Perform Cholesky decomposition: A = L @ L^T
    Returns L matrix plus step-by-step details.
    Raises ValueError if matrix is not SPD or not square/symmetric.
    """
    n = len(matrix)
    A = [row[:] for row in matrix]  # deep copy

    # Validate square
    for i, row in enumerate(A):
        if len(row) != n:
            raise ValueError("Matrix must be square.")

    # Validate symmetry (tolerance 1e-9)
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
        # ── Diagonal entry l_jj ─────────────────────────────────────────────
        sum_sq = sum(L[j][k] ** 2 for k in range(j))
        val = A[j][j] - sum_sq

        if val <= 1e-14:
            raise ValueError(
                f"Matrix is not positive-definite. "
                f"At column {j+1}: A[{j+1}][{j+1}] - Σ L²= {val:.8g} ≤ 0"
            )

        L[j][j] = math.sqrt(val)

        steps.append({
            "type": "diagonal",
            "row": j + 1,
            "col": j + 1,
            "formula": (
                f"L[{j+1}][{j+1}] = sqrt(A[{j+1}][{j+1}]"
                + "".join(f" - L[{j+1}][{k+1}]²" for k in range(j))
                + f") = sqrt({A[j][j]}"
                + (f" - {round(sum_sq,6)}" if j > 0 else "")
                + f") = sqrt({round(val,6)})"
            ),
            "sum_sq": round(sum_sq, 8),
            "val": round(val, 8),
            "result": round(L[j][j], 8),
        })

        # ── Sub-diagonal entries l_ij (i > j) ──────────────────────────────
        for i in range(j + 1, n):
            sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
            numerator = A[i][j] - sum_prod
            L[i][j] = numerator / L[j][j]

            steps.append({
                "type": "off_diagonal",
                "row": i + 1,
                "col": j + 1,
                "formula": (
                    f"L[{i+1}][{j+1}] = (A[{i+1}][{j+1}]"
                    + "".join(
                        f" - L[{i+1}][{k+1}]·L[{j+1}][{k+1}]" for k in range(j)
                    )
                    + f") / L[{j+1}][{j+1}]"
                    + f" = ({A[i][j]}"
                    + (f" - {round(sum_prod,6)}" if j > 0 else "")
                    + f") / {round(L[j][j],6)}"
                ),
                "sum_prod": round(sum_prod, 8),
                "numerator": round(numerator, 8),
                "divisor": round(L[j][j], 8),
                "result": round(L[i][j], 8),
            })

    # ── Verification: reconstruct A = L @ L^T ──────────────────────────────
    reconstructed = [
        [round(sum(L[r][k] * L[c][k] for k in range(n)), 6) for c in range(n)]
        for r in range(n)
    ]
    L_rounded = [[round(L[i][j], 6) for j in range(n)] for i in range(n)]

    return {"L": L_rounded, "steps": steps, "reconstructed": reconstructed, "n": n}


def solve_cholesky(L, b):
    """
    Given Cholesky factor L, solve Ax=b via:
      1. Forward substitution:  Ly  = b  → y
      2. Backward substitution: L^T x = y → x
    """
    n = len(L)

    # Forward: Ly = b
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][k] * y[k] for k in range(i))
        y[i] = (b[i] - s) / L[i][i]

    # Backward: L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / L[i][i]

    return [round(v, 6) for v in x], [round(v, 6) for v in y]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json(force=True)
        raw = data.get("matrix", [])
        solve = data.get("solve", False)
        b_vec = data.get("b", [])

        n = len(raw)
        if n < 2 or n > 6:
            return jsonify({"error": "Matrix size must be between 2×2 and 6×6."}), 400

        # Parse and validate entries
        matrix = []
        for i, row in enumerate(raw):
            if len(row) != n:
                return jsonify({"error": "Matrix must be square."}), 400
            parsed_row = []
            for j, val in enumerate(row):
                try:
                    parsed_row.append(float(val))
                except (TypeError, ValueError):
                    return jsonify(
                        {"error": f"Invalid value at row {i+1}, column {j+1}."}
                    ), 400
            matrix.append(parsed_row)

        result = cholesky_decompose(matrix)

        if solve and b_vec:
            if len(b_vec) != n:
                return jsonify(
                    {"error": "Vector b must have the same length as the matrix size."}
                ), 400
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
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
