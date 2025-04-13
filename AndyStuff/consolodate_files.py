import os
import shutil
import csv
from tkinter import Tk, filedialog
from tqdm import tqdm  # Make sure tqdm is installed: pip install tqdm

def consolidate_sessions_gui():
    # --- GUI Prompt for Folder Selection ---
    Tk().withdraw()
    sessions_output_folder = filedialog.askdirectory(title="Select Sessions Output Folder")

    if not sessions_output_folder:
        print("No folder selected. Exiting.")
        return

    # --- Output folder setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    consolidated_folder = os.path.join(script_dir, 'consolidated')
    txt_out_dir = os.path.join(consolidated_folder, 'text')
    xml_out_dir = os.path.join(consolidated_folder, 'xml')
    os.makedirs(txt_out_dir, exist_ok=True)
    os.makedirs(xml_out_dir, exist_ok=True)

    all_expressions_path = os.path.join(consolidated_folder, 'all_expressions.txt')
    mapping_csv_path = os.path.join(consolidated_folder, 'mapping.csv')

    # --- Gather all session folders ---
    sessions = [
        d for d in os.listdir(sessions_output_folder)
        if os.path.isdir(os.path.join(sessions_output_folder, d))
    ]

    # Count total text files first for progress bar
    total_files = 0
    for session in sessions:
        text_dir = os.path.join(sessions_output_folder, session, 'text')
        if os.path.isdir(text_dir):
            total_files += len([f for f in os.listdir(text_dir) if f.endswith('.txt')])

    index = 1
    with open(all_expressions_path, 'w', encoding='utf-8') as all_expressions_file, \
         open(mapping_csv_path, 'w', newline='', encoding='utf-8') as mapping_file:

        writer = csv.writer(mapping_file)
        writer.writerow(['index', 'text_filename', 'xml_filename', 'latex_expression'])

        with tqdm(total=total_files, desc="Consolidating files", unit="file") as pbar:
            for session in sessions:
                text_dir = os.path.join(sessions_output_folder, session, 'text')
                xml_dir = os.path.join(sessions_output_folder, session, 'xml')

                if not (os.path.isdir(text_dir) and os.path.isdir(xml_dir)):
                    continue

                text_files = sorted(f for f in os.listdir(text_dir) if f.endswith('.txt'))

                for txt_file in text_files:
                    txt_path = os.path.join(text_dir, txt_file)
                    xml_file = txt_file.replace('.txt', '.xml')
                    xml_path = os.path.join(xml_dir, xml_file)

                    if not os.path.exists(xml_path):
                        print(f"Warning: Missing XML file for {txt_file}")
                        pbar.update(1)
                        continue

                    new_base = f"{index:05d}"
                    new_txt = new_base + '.txt'
                    new_xml = new_base + '.xml'

                    shutil.copy(txt_path, os.path.join(txt_out_dir, new_txt))
                    shutil.copy(xml_path, os.path.join(xml_out_dir, new_xml))

                    with open(txt_path, 'r', encoding='utf-8') as f:
                        expression = f.read().strip().replace('−', '-')

                    all_expressions_file.write(expression + '\n')
                    writer.writerow([index, new_txt, new_xml, expression])
                    index += 1
                    pbar.update(1)

    print(f"\nConsolidation complete. Total expressions: {index - 1}")
    print(f"Output saved to: {consolidated_folder}")

if __name__ == "__main__":
    consolidate_sessions_gui()
