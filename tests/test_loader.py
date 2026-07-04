from rag_assistant.ingestion.loader import load_multiple_files


def test_load_multiple_files(tmp_path):
    temp_file0 = tmp_path / "non_existent_file.txt"
    temp_file1 = tmp_path / "dummy_file1.txt"
    temp_file2 = tmp_path / "dummy_file2.txt"
    temp_file3 = tmp_path / "dummy_file3.txt"

    expected_content = [
        "Content for dummy_file1.txt",
        "Content for dummy_file2.txt",
        "Content for dummy_file3.txt",
    ]
    temp_file1.write_text("Content for dummy_file1.txt")
    temp_file2.write_text("Content for dummy_file2.txt")
    temp_file3.write_text("Content for dummy_file3.txt")

    temp_test = load_multiple_files([temp_file0, temp_file1, temp_file2, temp_file3])

    assert temp_test == expected_content
