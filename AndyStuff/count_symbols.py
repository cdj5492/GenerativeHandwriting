import os
from collections import Counter
from tkinter import Tk, filedialog

# List of valid LaTeX symbols to count, sorted by longest first
allowed_symbols = [
    '\\frac{', '\\sqrt{', '\\neq', '\\geq', '\\leq',
    '\\times', '\\pi',
    '^{', '_{',
    '(', ')', '^', '_', '=', '+', '-', '>', '<',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

# Sort so longest symbols get matched first
allowed_symbols.sort(key=len, reverse=True)

def count_symbol_frequencies(filepath):
    counter = Counter()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            i = 0
            while i < len(line):
                if line[i] == ' ':
                    i += 1
                    continue

                matched = False
                for symbol in allowed_symbols:
                    if line[i:i+len(symbol)] == symbol:
                        counter[symbol] += 1
                        i += len(symbol)
                        matched = True
                        break

                if not matched:
                    i += 1  # skip unknown or irrelevant character

    # Ensure all symbols are present in the result
    for symbol in allowed_symbols:
        if symbol not in counter:
            counter[symbol] = 0

    return counter

def main():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select LaTeX expression file",
        filetypes=[("Text Files", "*.txt")]
    )

    if not file_path:
        print("No file selected.")
        return

    counts = count_symbol_frequencies(file_path)

    print("\nSymbol Frequencies:")
    for symbol in allowed_symbols:
        print(f"{symbol:8} : {counts[symbol]}")

if __name__ == "__main__":
    main()
