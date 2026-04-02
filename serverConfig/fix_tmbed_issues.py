import importlib
import os
import re
from pathlib import Path


def update_tmbed_package():
    try:
        tmbed = importlib.import_module("tmbed")
    except Exception as exc:
        return (
            "TMbed is not installed in this image; skipping TMbed patch. "
            "This is expected for non-ML services such as celery-worker or celery-beat "
            f"({exc})."
        )

    package_dir = Path(tmbed.__file__).resolve().parent
    patched_files = []

    for file_path in package_dir.rglob("*.py"):
        original = file_path.read_text()
        updated = original

        # Newer transformers uses `dtype` instead of `torch_dtype`.
        updated = updated.replace("torch_dtype=", "dtype=")

        # Make tokenizer behavior explicit so transformers stops warning
        # about the implicit default legacy setting.
        updated = re.sub(
            r"(T5Tokenizer\.from_pretrained\((?![^)]*legacy=)([^)]*))\)",
            r"\1, legacy=True)",
            updated,
            flags=re.DOTALL,
        )

        if updated != original:
            file_path.write_text(updated)
            patched_files.append(str(file_path))

    if not patched_files:
        return f"No TMbed patch changes were needed under {package_dir}."

    return (
        f"Patched {len(patched_files)} TMbed file(s) under {package_dir}: "
        + ", ".join(os.path.basename(path) for path in patched_files)
    )


if __name__ == "__main__":
    print(update_tmbed_package())
