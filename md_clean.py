import os
import re

# Define the root directory containing your .md files
ROOT_DIR = "data/"

# Function to check if the directory exists
def check_directory_exists(directory):
    """
    Checks if the specified directory exists.
    Exits the script if the directory does not exist.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        exit(1)
    elif not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        exit(1)

# Define your cleaning function
def clean_markdown(content):
    """
    Cleans the Markdown content based on predefined rules.
    Modify this function to include your specific cleaning logic.
    """
    # 1. Remove trailing whitespace from each line
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # 2. Standardize headings (ensure a space after #, ##, etc.)
    content = re.sub(r'^(#+)(\S)', r'\1 \2', content, flags=re.MULTILINE)
    
    # 3. Remove excessive blank lines (more than 2 consecutive blank lines)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 4. Remove HTML tags (optional, uncomment if needed)
    content = re.sub(r'<[^>]+>', '', content)
    
    # 5. Remove special characters or unwanted patterns (customize as needed)
    content = re.sub(r'[^a-zA-Z0-9\s\*\#\`\[\]\(\)\!\.\,\-\_]', '', content)
    
    # 6. Remove empty lines
    content = re.sub(r'^\s*$', '', content, flags=re.MULTILINE)
    
    return content

# Traverse the directory structure and clean .md files
def clean_all_md_files(root_dir):
    """
    Traverses the directory structure starting from `root_dir`,
    cleans all .md files, and saves them back in place.
    Returns the count of processed files.
    """
    file_count = 0  # Initialize a counter for processed files
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".md"):
                file_path = os.path.join(dirpath, filename)
                
                # Read the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                
                # Clean the content
                cleaned_content = clean_markdown(content)
                
                # Write the cleaned content back to the same file
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(cleaned_content)
                    file_count += 1  # Increment the counter for successfully processed files
                except Exception as e:
                    print(f"Error writing to {file_path}: {e}")
                    continue
    
    return file_count  # Return the total count of processed files

# Run the script
if __name__ == "__main__":
    print("Starting cleaning process...")
    
    # Check if the directory exists
    check_directory_exists(ROOT_DIR)
    
    # Clean all .md files and get the count of processed files
    processed_files_count = clean_all_md_files(ROOT_DIR)
    
    print(f"Cleaning completed successfully! Processed {processed_files_count} .md files.")