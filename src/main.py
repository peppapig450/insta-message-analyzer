import sys
from pathlib import Path

from core.data_loader import load_messages

# Ensure console output supports UTF-8 (Windows fix)
if sys.platform == "win32":
    import os
    os.system("chcp 65001")  # Set console to UTF-8

def main():
    # Resolve the project root (parent of src/) from main.py's location
    project_root = Path(__file__).parent.parent.resolve()
    root_dir = project_root / "data" / "your_instagram_activity" / "messages"
    output_dir = project_root / "output"
    output_path = output_dir / "messages_raw.csv"

    # Create output directory if it doesn’t exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading messages from: {root_dir}")
    df = load_messages(root_dir)

    if df.empty:
        print("No messages loaded. Check if 'inbox/' directory exists.")
    else:
        print("Loaded DataFrame with chat types:")
        print(df.head())
        print("\nChat type counts:")
        print(df["chat_type"].value_counts())
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved DataFrame to {output_path}")

if __name__ == "__main__":
    main()
