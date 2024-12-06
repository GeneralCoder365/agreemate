import os, json

def generate_symlink_mapping(directory):
    """Generate a mapping of symlink names to their target paths in a directory."""
    mappings = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.islink(filepath):
            target = os.path.abspath(os.path.join(directory, os.readlink(filepath)))
            mappings[filename] = target
    return mappings

def save_mappings_to_json(mappings, output_file):
    """Save the mappings to a JSON file, overwriting if the file already exists."""
    with open(output_file, "w") as f:
        json.dump(mappings, f, indent=4)
    print(f"Symlink mappings saved to: {output_file}")


if __name__ == "__main__":
    # directory containing the symlinks
    sym_dir = r"D:\Personal_Folders\Tocho\UMD\fall_2024\CMSC723\agreemate\finetuning\models--meta-llama--Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659"

    # generate the mappings
    symlink_mappings = generate_symlink_mapping(sym_dir)

    # path to output JSON file
    lowest_dirname = os.path.basename(os.path.normpath(sym_dir))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, f"{lowest_dirname}_symlink_mappings.json")

    # save mappings to JSON
    save_mappings_to_json(symlink_mappings, output_file)