"""
Parses Java source files and Markdown docs into structured chunks
suitable for embedding and retrieval.

Uses regex-based AST-aware extraction for Java (class boundaries,
method boundaries, Javadoc) and sliding-window for Markdown/XML.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CodeChunk:
    """A single semantic chunk extracted from source code or documentation."""

    id: str
    file_path: str
    chunk_type: str  # "class" | "method" | "file_summary" | "doc"
    content: str
    class_name: str = ""
    method_name: str = ""
    line_start: int = 0
    line_end: int = 0
    docstring: str = ""
    metadata: dict = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        """Returns the text that should be embedded for similarity search."""
        parts = []
        if self.class_name:
            parts.append(f"Class: {self.class_name}")
        if self.method_name:
            parts.append(f"Method: {self.method_name}")
        if self.docstring:
            parts.append(f"Description: {self.docstring}")
        parts.append(self.content)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "chunk_type": self.chunk_type,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "docstring": self.docstring,
        }


# ── Regex Patterns for Java Parsing ──────────────────────────────────

# Matches Javadoc comments: /** ... */
JAVADOC_PATTERN = re.compile(
    r"/\*\*\s*(.*?)\s*\*/", re.DOTALL
)

# Matches class/interface declarations
CLASS_PATTERN = re.compile(
    r"^(\s*(?:public|private|protected)?\s*(?:abstract|static|final)*\s*"
    r"(?:class|interface|enum|record)\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{)",
    re.MULTILINE,
)

# Matches method declarations (handles generics, annotations, throws)
METHOD_PATTERN = re.compile(
    r"^(\s*(?:@\w+(?:\(.*?\))?\s*)*"                    # annotations
    r"(?:public|private|protected)\s+"                    # access modifier
    r"(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?"   # optional modifiers
    r"(?:<[\w,\s?]+>\s+)?"                                # optional generics
    r"(?:[\w<>\[\],\s]+)\s+"                              # return type
    r"(\w+)\s*"                                           # method name
    r"\([^)]*\)"                                          # parameters
    r"(?:\s+throws\s+[\w,\s]+)?"                          # optional throws
    r"\s*\{)",                                            # opening brace
    re.MULTILINE,
)

# Matches single-line field declarations
FIELD_PATTERN = re.compile(
    r"^\s+(?:private|protected|public)\s+(?:static\s+)?(?:final\s+)?"
    r"(?:[\w<>\[\],\s]+)\s+(\w+)\s*[=;]",
    re.MULTILINE,
)


def _find_matching_brace(text: str, start: int) -> int:
    """Finds the position of the closing brace that matches the opening brace at `start`."""
    depth = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    i = start

    while i < len(text):
        c = text[i]
        next_c = text[i + 1] if i + 1 < len(text) else ""

        # Handle comment/string state transitions
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
        elif in_block_comment:
            if c == "*" and next_c == "/":
                in_block_comment = False
                i += 1
        elif in_string:
            if c == "\\" :
                i += 1  # skip escaped char
            elif c == '"':
                in_string = False
        elif in_char:
            if c == "\\":
                i += 1
            elif c == "'":
                in_char = False
        else:
            if c == "/" and next_c == "/":
                in_line_comment = True
                i += 1
            elif c == "/" and next_c == "*":
                in_block_comment = True
                i += 1
            elif c == '"':
                in_string = True
            elif c == "'":
                in_char = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    return len(text) - 1  # fallback


def _extract_preceding_javadoc(text: str, pos: int) -> str:
    """Extracts the Javadoc comment immediately preceding position `pos`."""
    # Look backwards from pos for a */ that closes a /** block
    search_region = text[max(0, pos - 2000) : pos].rstrip()
    match = re.search(r"/\*\*\s*(.*?)\s*\*/\s*$", search_region, re.DOTALL)
    if match:
        doc = match.group(1)
        # Clean up Javadoc formatting
        lines = doc.split("\n")
        cleaned = []
        for line in lines:
            line = re.sub(r"^\s*\*\s?", "", line).strip()
            if line and not line.startswith("@"):
                cleaned.append(line)
        return " ".join(cleaned)
    return ""


def _line_number_at(text: str, pos: int) -> int:
    """Returns the 1-indexed line number at a character position."""
    return text[:pos].count("\n") + 1


def parse_java_file(file_path: Path) -> list[CodeChunk]:
    """
    Parses a single Java file into semantic chunks.
    Returns class-level and method-level chunks.
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(file_path).replace("\\", "/")
    chunks: list[CodeChunk] = []

    # ── Extract package and imports as context ─────────────────
    package_match = re.search(r"^package\s+([\w.]+);", content, re.MULTILINE)
    package_name = package_match.group(1) if package_match else ""

    import_lines = re.findall(r"^import\s+[\w.*]+;", content, re.MULTILINE)
    imports_text = "\n".join(import_lines[:15])  # cap at 15 imports

    # ── Extract class-level chunks ─────────────────────────────
    for class_match in CLASS_PATTERN.finditer(content):
        class_name = class_match.group(2)
        class_start = class_match.start()
        brace_pos = content.index("{", class_match.start())
        class_end = _find_matching_brace(content, brace_pos)

        class_body = content[class_start : class_end + 1]
        class_doc = _extract_preceding_javadoc(content, class_start)

        # Collect field names for class summary
        field_matches = FIELD_PATTERN.findall(class_body)
        fields_summary = ", ".join(field_matches[:10]) if field_matches else ""

        class_summary_parts = [
            f"package {package_name};" if package_name else "",
            imports_text,
            "",
            f"// Fields: {fields_summary}" if fields_summary else "",
            class_body[:500] + ("..." if len(class_body) > 500 else ""),
        ]

        chunks.append(
            CodeChunk(
                id=f"{class_name}.__class__",
                file_path=rel_path,
                chunk_type="class",
                class_name=class_name,
                content="\n".join(p for p in class_summary_parts if p),
                line_start=_line_number_at(content, class_start),
                line_end=_line_number_at(content, class_end),
                docstring=class_doc,
                metadata={"package": package_name},
            )
        )

        # ── Extract method-level chunks within this class ──────
        class_content = content[class_start : class_end + 1]
        for method_match in METHOD_PATTERN.finditer(class_content):
            method_name = method_match.group(2)
            method_start_in_class = method_match.start()
            method_abs_start = class_start + method_start_in_class

            # Find the opening brace of the method
            brace_pos_method = class_content.index("{", method_start_in_class)
            method_end_in_class = _find_matching_brace(
                class_content, brace_pos_method
            )
            method_abs_end = class_start + method_end_in_class

            method_body = class_content[method_start_in_class : method_end_in_class + 1]
            method_doc = _extract_preceding_javadoc(
                content, method_abs_start
            )

            chunks.append(
                CodeChunk(
                    id=f"{class_name}.{method_name}",
                    file_path=rel_path,
                    chunk_type="method",
                    class_name=class_name,
                    method_name=method_name,
                    content=method_body,
                    line_start=_line_number_at(content, method_abs_start),
                    line_end=_line_number_at(content, method_abs_end),
                    docstring=method_doc,
                    metadata={"package": package_name},
                )
            )

    # If no classes were found, treat the whole file as one chunk
    if not chunks:
        chunks.append(
            CodeChunk(
                id=file_path.stem,
                file_path=rel_path,
                chunk_type="file_summary",
                content=content[:2000],
                line_start=1,
                line_end=content.count("\n") + 1,
            )
        )

    return chunks


def parse_markdown_file(file_path: Path, chunk_size: int = 1500) -> list[CodeChunk]:
    """
    Parses a Markdown file into section-based chunks.
    Splits on ## headings and caps chunk size.
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(file_path).replace("\\", "/")
    chunks: list[CodeChunk] = []

    # Split by ## headings
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract heading
        heading_match = re.match(r"^##?\s+(.+)", section)
        heading = heading_match.group(1).strip() if heading_match else f"Section {i}"

        # Split large sections into sub-chunks
        if len(section) > chunk_size:
            for j in range(0, len(section), chunk_size - 200):
                sub = section[j : j + chunk_size]
                chunks.append(
                    CodeChunk(
                        id=f"{file_path.stem}.{heading}.part{j // chunk_size}",
                        file_path=rel_path,
                        chunk_type="doc",
                        content=sub,
                        docstring=heading,
                        metadata={"doc_type": "markdown"},
                    )
                )
        else:
            chunks.append(
                CodeChunk(
                    id=f"{file_path.stem}.{heading}",
                    file_path=rel_path,
                    chunk_type="doc",
                    content=section,
                    docstring=heading,
                    metadata={"doc_type": "markdown"},
                )
            )

    return chunks


def parse_pom_file(file_path: Path) -> list[CodeChunk]:
    """Extracts key information from pom.xml as a single chunk."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(file_path).replace("\\", "/")

    return [
        CodeChunk(
            id="pom.xml",
            file_path=rel_path,
            chunk_type="doc",
            content=content[:3000],
            docstring="Maven POM — project dependencies, build plugins, and Java 21 config",
            metadata={"doc_type": "pom"},
        )
    ]


def parse_codebase(source_dir: Path, project_root: Path) -> list[CodeChunk]:
    """
    Walks the entire codebase and returns all chunks.

    Args:
        source_dir: Path to the Java source root (com/drivestream/)
        project_root: Path to the project root (for README, pom.xml, etc.)

    Returns:
        List of all CodeChunk objects ready for embedding.
    """
    all_chunks: list[CodeChunk] = []

    # Parse Java source files
    for java_file in sorted(source_dir.rglob("*.java")):
        try:
            file_chunks = parse_java_file(java_file)
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"  ⚠ Failed to parse {java_file.name}: {e}")

    # Parse Markdown documentation
    for md_file in ["README.md", "SETUP_GUIDE.md"]:
        md_path = project_root / md_file
        if md_path.exists():
            try:
                all_chunks.extend(parse_markdown_file(md_path))
            except Exception as e:
                print(f"  ⚠ Failed to parse {md_file}: {e}")

    # Parse pom.xml
    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            all_chunks.extend(parse_pom_file(pom_path))
        except Exception as e:
            print(f"  ⚠ Failed to parse pom.xml: {e}")

    return all_chunks
