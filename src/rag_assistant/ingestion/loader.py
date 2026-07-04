def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_multiple_files(paths):
    contents = []
    for path in paths:
        try:
            content = load_file(path)
            contents.append(content)
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue
    return contents
