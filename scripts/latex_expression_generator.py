#!/usr/bin/env python3
"""
Probabilistic LaTeX expression generator
---------------------------------------

Mixes    1) simple templated exercises (linear, quadratic, basic fractions)  
and      2) a grammar whose choices are *weighted* by empirical token
          frequencies (letters flattened to equal likelihood).

Only the original allowed tokens are ever emitted.
"""

import argparse
import random
from typing import List

# ---------------------------------------------------------------------------
#  Empirical probabilities (from your corpus, letters flattened)
# ---------------------------------------------------------------------------
TOKEN_PROB = {
    "\\infty": 0.000218,
    "\\ldots": 0.000163,
    "\\times ": 0.003592,
    "\\theta": 0.000925,
    "\\alpha": 0.003429,
    "\\gamma": 0.000272,
    "\\lambda": 0.000272,
    "\\sigma": 0.000218,
    "\\cdot": 0.000163,
    "\\frac{": 0.016928,
    "\\sqrt{": 0.002558,
    "\\log_": 0.002341,
    "\\neq": 0.000653,
    "\\beta": 0.00049,
    "\\phi": 0.000381,
    "\\div": 0.000871,
    "\\geq": 0.001089,
    "\\leq": 0.001415,
    "\\sin": 0.00332,
    "\\cos": 0.002558,
    "\\tan": 0.002394,
    "\\mu": 0.000544,
    "\\pi": 0.000816,
    "\\pm": 0.00049,
    "^{": 0.001144,
    "_{": 0.001033,
    "(": 0.011659,
    ")": 0.011659,
    "{": 0.022636,
    "}": 0.022636,
    "^": 0.002504,
    "_": 0.002667,
    "=": 0.011548,
    "+": 0.016655,
    "-": 0.017741,
    "!": 0.000109,
    ">": 0.004033,
    "<": 0.004251,
    " ": 0.124122,
    "0": 0.005228,
    "1": 0.007407,
    "2": 0.007135,
    "3": 0.006972,
    "4": 0.005871,
    "5": 0.005488,
    "6": 0.005543,
    "7": 0.006102,
    "8": 0.005716,
    "9": 0.006212,
    "A": 0.003542,
    "B": 0.003542,
    "C": 0.003542,
    "D": 0.003542,
    "E": 0.003542,
    "F": 0.003542,
    "G": 0.003542,
    "H": 0.003542,
    "L": 0.003542,
    "M": 0.003542,
    "N": 0.003542,
    "P": 0.003542,
    "R": 0.003542,
    "S": 0.003542,
    "T": 0.003542,
    "V": 0.003542,
    "X": 0.003542,
    "Y": 0.003542,
    "a": 0.003542,
    "b": 0.003542,
    "c": 0.003542,
    "d": 0.003542,
    "e": 0.003542,
    "f": 0.003542,
    "g": 0.003542,
    "h": 0.003542,
    "i": 0.003542,
    "j": 0.003542,
    "k": 0.003542,
    "l": 0.003542,
    "m": 0.003542,
    "n": 0.003542,
    "o": 0.003542,
    "p": 0.003542,
    "q": 0.003542,
    "r": 0.003542,
    "s": 0.003542,
    "t": 0.003542,
    "u": 0.003542,
    "v": 0.003542,
    "w": 0.003542,
    "x": 0.003542,
    "y": 0.003542,
    "z": 0.003542
}

TOKENS = list(TOKEN_PROB)

# pools
LETTERS   = [t for t in TOKENS if t.isalpha()]
DIGITS    = list("0123456789")
ADV_FUN   = ["\\sin", "\\cos", "\\tan", "\\log_"]
ADV_OPS   = ["\\times ", "\\cdot", "\\div"]
REL_OPS   = ["=", "\\neq", "\\geq", "\\leq", ">", "<"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def w(tok_list: List[str]) -> str:
    weights = [TOKEN_PROB[t] for t in tok_list]
    return random.choices(tok_list, weights)[0]

def rand_coeff(lo=1, hi=10, neg=True) -> int:
    k = random.randint(lo, hi)
    return -k if neg and random.random() < 0.4 else k

def var() -> str:
    return w(LETTERS)

# ---------------------------------------------------------------------------
# template mode
# ---------------------------------------------------------------------------
def template_expr() -> str:
    roll = random.random()
    if roll < 0.5:
        # a(bx+c)+d=e
        a, b, c, d = [rand_coeff() for _ in range(4)]
        v   = var()
        lhs = f"{a}({b}{v}{'+' if c>=0 else ''}{c})"
        if random.random() < 0.5:
            lhs += f"{random.choice(['+','-'])}{abs(rand_coeff())}"
        rhs = str(d)
        return lhs + "=" + rhs
    elif roll < 0.8:
        # \frac{ax+b}{c}=d
        v = var()
        a, b, c, d = rand_coeff(), rand_coeff(), random.randint(1,9), rand_coeff(-20,20,True)
        return f"\\frac{{{a}{v}+{b}}}{{{c}}}={d}"
    else:
        # ax^2+bx+c=0
        a, b, c = rand_coeff(1,9,False), rand_coeff(), rand_coeff()
        v = var()
        return f"{a}{v}^2{'+-'[b<0]}{abs(b)}{v}{'+-'[c<0]}{abs(c)}=0"

# ---------------------------------------------------------------------------
# grammar mode (uses weights everywhere)
# ---------------------------------------------------------------------------
MAX_DEPTH = 3

def atom(adv: bool) -> str:
    base = LETTERS + DIGITS
    if adv:
        base += ["\\infty", "\\pi"]
    return w(base)

def maybe_power(expr: str, adv: bool) -> str:
    if random.random() < 0.25:
        op = w(["^", "_"])
        if adv and random.random() < 0.6:
            expr += op + "{" + atom(adv) + "}"
        else:
            expr += op + atom(adv)
    return expr

def factor(adv: bool, d=0) -> str:
    if d >= MAX_DEPTH:
        return maybe_power(atom(adv), adv)
    r = random.random()
    if adv and r < 0.15:
        return "\\sqrt{" + term(adv, d+1) + "}"
    if adv and r < 0.3:
        return "\\frac{" + term(adv, d+1) + "}{" + term(adv, d+1) + "}"
    if adv and r < 0.55:
        f = w(ADV_FUN)
        if f == "\\log_":
            return "\\log_{" + atom(adv) + "}(" + term(adv, d+1) + ")"
        return f + "(" + term(adv, d+1) + ")"
    if r < 0.7:
        return "(" + expr(adv, d+1, 6) + ")"
    return maybe_power(atom(adv), adv)

def term(adv: bool, d=0) -> str:
    parts = [factor(adv, d) for _ in range(random.randint(1,3))]
    out   = parts[0]
    for p in parts[1:]:
        if adv and random.random() < 0.5:
            out += w(ADV_OPS)
        out += p
    return out

def expr(adv: bool, d=0, target=20) -> str:
    e = term(adv, d)
    while len(e) < target and random.random() < 0.5:
        e += w(["+","-"]) + term(adv, d)
    return e

def grammar_line(eq_p: float, adv: bool, mean_len: int) -> str:
    left = expr(adv, 0, mean_len)
    if random.random() < eq_p:
        rel   = w(REL_OPS if adv else ["="])
        right = expr(adv, 0, mean_len)
        return left + rel + right
    return left

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--num-expr", type=int, default=1000)
    ap.add_argument("--eq-prob", type=float, default=0.4)
    ap.add_argument("--basic-only-prob", type=float, default=0.3)
    ap.add_argument("--template-prob", type=float, default=0.6)
    ap.add_argument("--mean-len", type=int, default=35)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out = []
    for _ in range(args.num_expr):
        use_template = random.random() < args.template_prob
        adv          = not (random.random() < args.basic_only_prob)
        line = template_expr() if use_template else grammar_line(
            args.eq_prob, adv, args.mean_len)
        out.append(line)

    with open(args.outfile, "w", encoding="utf8") as fh:
        fh.write("\n".join(out))
    print("Wrote", len(out), "expressions to", args.outfile)

if __name__ == "__main__":
    main()
