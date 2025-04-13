import os
from tkinter import Tk, filedialog
from collections import Counter

def count_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return Counter(text)

def main():
    # Open GUI for selecting a text file
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a .txt file",
        filetypes=[("Text files", "*.txt")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    char_counts = count_characters(file_path)

    # Sort by character code for readability
    sorted_counts = sorted(char_counts.items(), key=lambda x: ord(x[0]))

    print("\nCharacter Frequency Table:\n")
    for char, count in sorted_counts:
        display_char = repr(char) if char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' else char
        print(f"{display_char}: {count}")

if __name__ == "__main__":
    main()
