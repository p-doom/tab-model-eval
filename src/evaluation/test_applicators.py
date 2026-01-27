import pytest
from .prediction_applicators import (
    parse_sed_command,
    apply_sed_to_content,
    apply_sed_prediction,
    parse_zeta_output,
    apply_zeta_prediction,
)


class TestSedParsing:
    def test_parse_delete_single_line(self):
        cmd = "sed -i '5d' file.py"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "delete"
        assert result["start_line"] == 5
        assert result["end_line"] == 5
        assert result["file_path"] == "file.py"

    def test_parse_delete_range(self):
        cmd = "sed -i '10,15d' src/main.py"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "delete"
        assert result["start_line"] == 10
        assert result["end_line"] == 15
        assert result["file_path"] == "src/main.py"

    def test_parse_replace_single_line(self):
        cmd = "sed -i '3c\\new content here' test.txt"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "replace"
        assert result["start_line"] == 3
        assert result["end_line"] == 3
        assert "new content" in result["content"]

    def test_parse_replace_range(self):
        cmd = "sed -i '1,5c\\replaced text' file.py"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "replace"
        assert result["start_line"] == 1
        assert result["end_line"] == 5

    def test_parse_insert(self):
        cmd = "sed -i '10i\\inserted line' file.py"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "insert"
        assert result["start_line"] == 10
        assert "inserted" in result["content"]

    def test_parse_append(self):
        cmd = "sed -i '$a\\appended content' file.py"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "append"
        assert result["start_line"] == -1
        assert "appended" in result["content"]

    def test_parse_with_bash_block(self):
        cmd = "```bash\nsed -i '5d' file.py\n```"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "delete"

    def test_parse_with_chain(self):
        cmd = "sed -i '5,10c\\new\nlines' file.py && cat -n file.py"
        result = parse_sed_command(cmd)
        assert result is not None
        assert result["operation"] == "replace"

    def test_invalid_command(self):
        assert parse_sed_command("echo hello") is None
        assert parse_sed_command("cat file.py") is None


class TestSedApplication:
    def test_delete_single_line(self):
        content = "line1\nline2\nline3\nline4"
        parsed = {"operation": "delete", "start_line": 2, "end_line": 2}
        result = apply_sed_to_content(content, parsed)
        assert result == "line1\nline3\nline4"

    def test_delete_range(self):
        content = "line1\nline2\nline3\nline4\nline5"
        parsed = {"operation": "delete", "start_line": 2, "end_line": 4}
        result = apply_sed_to_content(content, parsed)
        assert result == "line1\nline5"

    def test_replace_single_line(self):
        content = "line1\nline2\nline3"
        parsed = {"operation": "replace", "start_line": 2, "end_line": 2, "content": "new line"}
        result = apply_sed_to_content(content, parsed)
        assert result == "line1\nnew line\nline3"

    def test_replace_with_multiline(self):
        content = "line1\nline2\nline3"
        parsed = {"operation": "replace", "start_line": 2, "end_line": 2, "content": "new1\nnew2"}
        result = apply_sed_to_content(content, parsed)
        assert result == "line1\nnew1\nnew2\nline3"

    def test_insert(self):
        content = "line1\nline2\nline3"
        parsed = {"operation": "insert", "start_line": 2, "end_line": 2, "content": "inserted"}
        result = apply_sed_to_content(content, parsed)
        assert result == "line1\ninserted\nline2\nline3"

    def test_append(self):
        content = "line1\nline2"
        parsed = {"operation": "append", "start_line": -1, "end_line": -1, "content": "appended"}
        result = apply_sed_to_content(content, parsed)
        assert result == "line1\nline2\nappended"


class TestSedPrediction:
    def test_apply_prediction(self):
        files = {"test.py": "line1\nline2\nline3"}
        prediction = "```bash\nsed -i '2c\\modified' test.py\n```"
        result, error = apply_sed_prediction(files, prediction)
        assert error is None
        assert result["test.py"] == "line1\nmodified\nline3"

    def test_file_not_found(self):
        files = {"other.py": "content"}
        prediction = "sed -i '1d' missing.py"
        result, error = apply_sed_prediction(files, prediction)
        assert error is not None
        assert "file_not_found" in error

    def test_partial_path_match(self):
        files = {"src/lib/test.py": "line1\nline2"}
        prediction = "sed -i '1d' test.py"
        result, error = apply_sed_prediction(files, prediction)
        assert error is None
        assert result["src/lib/test.py"] == "line2"


class TestZetaParsing:
    def test_parse_basic_output(self):
        output = """<output>
```src/main.rs
<|editable_region_start|>
fn main() {
    println!("Hello");
}
<|editable_region_end|>
```
</output>"""
        result = parse_zeta_output(output)
        assert result is not None
        assert "src/main.rs" in result
        assert "fn main()" in result["src/main.rs"]
        assert "<|editable_region" not in result["src/main.rs"]

    def test_parse_without_output_tags(self):
        output = """```test.py
def hello():
    pass
```"""
        result = parse_zeta_output(output)
        assert result is not None
        assert "test.py" in result

    def test_removes_cursor_marker(self):
        output = """<output>
```file.py
line1<|user_cursor_is_here|>
line2
```
</output>"""
        result = parse_zeta_output(output)
        assert result is not None
        assert "<|user_cursor_is_here|>" not in result["file.py"]

    def test_multiple_files(self):
        output = """<output>
```file1.py
content1
```
```file2.py
content2
```
</output>"""
        result = parse_zeta_output(output)
        assert result is not None
        assert len(result) == 2
        assert "file1.py" in result
        assert "file2.py" in result


class TestZetaPrediction:
    def test_apply_prediction(self):
        files = {"src/main.rs": "old content"}
        prediction = """<output>
```src/main.rs
new content
```
</output>"""
        result, error = apply_zeta_prediction(files, prediction)
        assert error is None
        assert result["src/main.rs"] == "new content\n"

    def test_partial_path_match(self):
        files = {"project/src/main.rs": "old"}
        prediction = """<output>
```src/main.rs
new
```
</output>"""
        result, error = apply_zeta_prediction(files, prediction)
        assert error is None
        assert result["project/src/main.rs"] == "new\n"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
