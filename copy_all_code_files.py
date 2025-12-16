import os
import shutil
from pathlib import Path

# Set your project root and the output folder
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_FOLDER = PROJECT_ROOT / "all_code_files"

# List of folders/files to include (relative to project root)
INCLUDE_FOLDERS = [
    "src",
    "notebooks",
    "app_new.py",
    # Add more if needed
]

# File extensions to include
INCLUDE_EXTENSIONS = {".py", ".ipynb", ".yaml", ".yml", ".md", ".txt"}


def clear_output_folder():
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def copy_files():
    for item in INCLUDE_FOLDERS:
        src_path = PROJECT_ROOT / item
        if src_path.is_file():
            # Copy single file
            if src_path.suffix in INCLUDE_EXTENSIONS:
                shutil.copy2(src_path, OUTPUT_FOLDER / src_path.name)
        elif src_path.is_dir():
            for root, _, files in os.walk(src_path):
                for file in files:
                    file_path = Path(root) / file
                    # Always include __init__.py, even if empty
                    if file == "__init__.py" or file_path.suffix in INCLUDE_EXTENSIONS:
                        # Handle __init__.py renaming
                        if file == "__init__.py":
                            parent_name = file_path.parent.name
                            new_name = f"{parent_name}_init.py"
                        else:
                            new_name = file
                        rel_dir = Path(root).relative_to(PROJECT_ROOT)
                        # To avoid name collisions, prefix with relative path
                        out_name = f"{str(rel_dir).replace(os.sep, '_')}_{new_name}" if rel_dir != Path('.') else new_name
                        shutil.copy2(file_path, OUTPUT_FOLDER / out_name)


def main():
    clear_output_folder()
    copy_files()
    print(f"✅ All code files copied to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
