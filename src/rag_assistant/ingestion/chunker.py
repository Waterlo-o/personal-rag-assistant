def find_nearest_char(text, position, radius=100, target_char="."):
    start = max(0, position - radius)
    end = min(len(text), position + radius + 1)
    window = text[start:end]

    abs_indices = []
    for local_index, char in enumerate(window):
        if char == target_char:
            abs_indices.append(start + local_index)

    if not abs_indices:
        if radius >= len(text):
            return None

        return find_nearest_char(text, position, radius + 100, target_char=target_char)

    nearest_index = min(abs_indices, key=lambda x: abs(x - position))
    return nearest_index


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    position = chunk_size
    while position < len(text):
        chunk_end = find_nearest_char(text, position, radius=100)
        if chunk_end is None:
            chunk_end = position
        inner_text = text[start : chunk_end + 1]
        chunks.append(inner_text)
        start = find_nearest_char(
            text, chunk_end - overlap, radius=100, target_char=" "
        )
        if start is None or start >= chunk_end:
            start = chunk_end + 1
        position = start + chunk_size
    if start < len(text):
        chunks.append(text[start:])
    return chunks
