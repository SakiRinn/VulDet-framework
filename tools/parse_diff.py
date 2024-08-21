import requests
from unidiff import PatchSet
from io import StringIO


def extract_original_functions(patch_set):
    original_functions = []

    for patched_file in patch_set:
        if patched_file.is_modified_file:
            for hunk in patched_file:
                for line in hunk.source_lines():
                    if line.is_added:  # Skip added lines
                        continue
                    if line.is_context:
                        continue

                    # Extract the function name and lines based on the context
                    line_number = line.source_line_no - 1  # Adjust for 0-index
                    start_line = max(0, line_number - 5)  # Assume a function starts 5 lines before change
                    function_lines = original_file_content[start_line:line_number + 5]

                    original_functions.append((patched_file.path, line_number, function_lines))

    return original_functions

original_functions = extract_original_functions(patch_set)

for file_name, line_number, function_lines in original_functions:
    print(f"File: {file_name}, Line: {line_number + 1}")
    print("Original Function:")
    print("\n".join(function_lines))
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    GITHUB_TOKEN = ''

    commit_sha = '70976a7926b42d87e0c575412b85a8f5c1e48fad'
    owner = 'qemu'
    repo = 'qemu'

    url = f'https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}'

    headers = {'Accept': 'application/vnd.github.v3.diff'}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Failed to fetch the commit diff.")
        exit()
    diff_content = response.text
    patch_set = PatchSet(StringIO(diff_content))
