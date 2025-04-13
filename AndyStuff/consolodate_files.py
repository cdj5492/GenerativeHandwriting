import os
import shutil
import csv
from tkinter import Tk, filedialog

def consolidate_sessions_gui():
    # --- GUI Prompt for Folder Selection ---
    Tk().withdraw()  # Hide root window
    sessions_output_folder = filedialog.askdirectory(title="Select Sessions Output Folder")

    if not sessions_output_folder:
        print("No folder selected. Exiting.")
        return

    # --- Output folder in the same directory as this script ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    consolidated_folder = os.path.join(script_dir, 'consolidated')
    os.makedirs(consolidated_folder, exist_ok=True)

    txt_out_dir = os.path.join(consolidated_folder, 'text')
    xml_out_dir = os.path.join(consolidated_folder, 'xml')
    os.makedirs(txt_out_dir, exist_ok=True)
    os.makedirs(xml_out_dir, exist_ok=True)

    all_expressions_path = os.path.join(consolidated_folder, 'all_expressions.txt')
    mapping_csv_path = os.path.join(consolidated_folder, 'mapping.csv')

    index = 1
    with open(all_expressions_path, 'w', encoding='utf-8') as all_expressions_file, \
         open(mapping_csv_path, 'w', newline='', encoding='utf-8') as mapping_file:

        writer = csv.writer(mapping_file)
        writer.writerow(['index', 'text_filename', 'xml_filename', 'latex_expression'])

        for root, dirs, _ in os.walk(sessions_output_folder):
            for session in dirs:
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

    print(f"\nConsolidation complete. Total expressions: {index - 1}")
    print(f"Output saved to: {consolidated_folder}")

# Run the function
if __name__ == "__main__":
    consolidate_sessions_gui()
