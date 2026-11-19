import os
import shutil
import sys

def remove_pycache(root_dir="."):
    """
    Recursively finds and deletes all '__pycache__' directories
    starting from root_dir.
    """
    root_path = os.path.abspath(root_dir)
    if not os.path.isdir(root_path):
        print(f"Error: Directory '{root_path}' does not exist.")
        return

    print(f"Scanning for '__pycache__' in: {root_path}\n")
    removed_count = 0

    # Walk from bottom up so child directories are removed before parent directories
    for dirpath, dirnames, _ in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            if dirname == "__pycache__":
                target = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(target)
                    removed_count += 1
                    print(f"Removed: {target}")
                except Exception as e:
                    print(f"Failed to remove {target}: {e}")

    print(f"\nFinished. Removed {removed_count} '__pycache__' folder(s).")

if __name__ == "__main__":
    # Target directory defaults to current directory, or takes an optional command-line argument
    target_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    remove_pycache(target_directory)