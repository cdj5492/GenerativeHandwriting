import os
from tkinter import Tk, filedialog

def generate_latex_document(lines):
    header = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}
\begin{document}
"""
    body = "\n".join([f"\\[\n{line.strip()}\n\\]" for line in lines if line.strip()])
    footer = r"\end{document}"
    return f"{header}\n{body}\n{footer}"

def main():
    Tk().withdraw()

    file_path = filedialog.askopenfilename(
        title="Select LaTeX expressions file (.txt)",
        filetypes=[("Text Files", "*.txt")]
    )

    if not file_path:
        print("No file selected.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        expressions = f.readlines()

    latex_code = generate_latex_document(expressions)

    # Save to .tex file
    output_path = os.path.join(os.path.dirname(file_path), "generated_latex_document.tex")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write(latex_code)

    print(f"✓ LaTeX document written to:\n{output_path}")

if __name__ == "__main__":
    main()
