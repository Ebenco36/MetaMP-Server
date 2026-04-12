import importlib
import importlib.metadata
import os
import re
from pathlib import Path


def _transformers_prefers_dtype() -> bool:
    try:
        version_text = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return False

    match = re.match(r"^(\d+)\.(\d+)", version_text)
    if not match:
        return False

    major = int(match.group(1))
    minor = int(match.group(2))

    # Older transformers releases still expect `torch_dtype=` in TMbed's
    # from_pretrained calls. Newer releases moved toward `dtype=`.
    return (major, minor) >= (4, 56)


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
    prefers_dtype = _transformers_prefers_dtype()
    dtype_keyword = "dtype=" if prefers_dtype else "torch_dtype="

    for file_path in package_dir.rglob("*.py"):
        original = file_path.read_text()
        updated = original

        # First normalize any previously broken repeated prefixes such as
        # `torch_torch_dtype=` back to the standard PyTorch spelling.
        updated = re.sub(r"(?:torch_)+dtype=", "dtype=", updated)

        # Align only `from_pretrained(...)` keyword spelling with the
        # installed transformers release. Plain PyTorch calls such as
        # `torch.tensor(..., dtype=...)` must remain untouched.
        updated = re.sub(
            r"(from_pretrained\([^\)]*?)torch_dtype=",
            rf"\1{dtype_keyword}",
            updated,
            flags=re.DOTALL,
        )
        updated = re.sub(
            r"(from_pretrained\([^\)]*?)(?<!torch_)dtype=",
            rf"\1{dtype_keyword}",
            updated,
            flags=re.DOTALL,
        )

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
        return (
            f"No TMbed patch changes were needed under {package_dir}. "
            f"transformers expects `{dtype_keyword[:-1]}`."
        )

    return (
        f"Patched {len(patched_files)} TMbed file(s) under {package_dir} "
        f"for transformers keyword `{dtype_keyword[:-1]}`: "
        + ", ".join(os.path.basename(path) for path in patched_files)
    )


if __name__ == "__main__":
    print(update_tmbed_package())
