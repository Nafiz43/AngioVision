from pathlib import Path

def write_dir_tree_to_txt(base_dir, output_txt):
    base_dir = Path(base_dir).resolve()
    output_txt = Path(output_txt).resolve()

    def build_lines(path, indent=0):
        lines = []
        prefix = "  " * indent

        if path.is_dir():
            lines.append(f"{prefix}{path.name}/")
            entries = sorted(
                path.iterdir(),
                key=lambda x: (x.is_file(), x.name.lower())
            )
            for p in entries:
                lines.extend(build_lines(p, indent + 1))
        else:
            lines.append(f"{prefix}{path.name}")

        return lines

    lines = build_lines(base_dir)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Directory structure saved to: {output_txt}")


# ======================
# Example usage
# ======================
if __name__ == "__main__":
    BASE_DIR = "/data/Deep_Angiography/DICOM_Sequence_Processed/"
    OUTPUT_TXT = "/data/Deep_Angiography/dir_structure.txt"

    write_dir_tree_to_txt(BASE_DIR, OUTPUT_TXT)
