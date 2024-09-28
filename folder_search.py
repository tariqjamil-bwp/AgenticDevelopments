def find_set_ais_utils_path(search_folder):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        possible_path = os.path.join(current_dir, search_folder)
        if os.path.isdir(possible_path):
            sys.path.append(possible_path)
            return possible_path  # Return the found path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"{search_folder} folder not found in the hierarchy")
        current_dir = parent_dir

# Use the function to find 'ais_utils' and set the path
utils_path = find_set_ais_utils_path('utils')
print(f"'utils' folder found at: {utils_path}")
