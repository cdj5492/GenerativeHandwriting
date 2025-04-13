import os
from tkinter import Tk, filedialog

def main():
    Tk().withdraw()

    # Select the file
    file_path = filedialog.askopenfilename(
        title="Select a file to deduplicate",
        filetypes=[("Text Files", "*.txt")]
    )

    if not file_path:
        print("No file selected.")
        return

    # Read and deduplicate lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Strip and deduplicate while preserving order
    seen = set()
    deduplicated = []
    for line in lines:
        stripped = line.strip()
        if stripped not in seen:
            seen.add(stripped)
            deduplicated.append(stripped)

    # Save to new file
    output_path = os.path.join(os.path.dirname(file_path), "deduplicated_output.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in deduplicated:
            f.write(line + "\n")

    print(f"\nDeduplicated output written to:\n{output_path}")

if __name__ == "__main__":
    main()
