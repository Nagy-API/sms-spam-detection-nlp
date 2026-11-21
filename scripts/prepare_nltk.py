"""Install the exact NLTK resources used for inference at build time."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from shutil import rmtree
from sysconfig import get_path
from urllib.request import urlopen
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_DATA_URL = f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_REVISION}/packages"
RESOURCES = {
    "corpora/stopwords.zip": "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
    "corpora/wordnet.zip": "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
    "tokenizers/punkt_tab.zip": "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
    "taggers/averaged_perceptron_tagger_eng.zip": (
        "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b"
    ),
}


def prepare_nltk():
    for archive, expected_digest in RESOURCES.items():
        with urlopen(f"{NLTK_DATA_URL}/{archive}", timeout=60) as response:
            content = response.read()
        if sha256(content).hexdigest() != expected_digest:
            raise ValueError(f"Unexpected checksum for NLTK resource {archive}")

        destination = NLTK_DATA_DIR / archive
        destination.parent.mkdir(parents=True, exist_ok=True)
        if archive == "corpora/wordnet.zip":
            destination.write_bytes(content)
        else:
            with ZipFile(BytesIO(content)) as package:
                members = package.namelist()
                if archive == "tokenizers/punkt_tab.zip":
                    members = [
                        name for name in members
                        if name.startswith("punkt_tab/english/") or name == "punkt_tab/README"
                    ]
                package.extractall(destination.parent, members=members)
        print(f"Prepared {archive}", flush=True)


def trim_dependency_tests():
    site_packages = Path(get_path("purelib")).resolve()
    if not site_packages.is_relative_to(PROJECT_ROOT / ".vercel" / "python"):
        raise RuntimeError("Dependency trimming requires Vercel's isolated build environment.")

    # These directories contain upstream test suites and fixtures, separate from library code.
    for package in ("numpy", "scipy", "sklearn"):
        for directory in (site_packages / package).rglob("tests"):
            if directory.is_dir() and directory.resolve().is_relative_to(site_packages / package):
                rmtree(directory)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--vercel-build", action="store_true", help="Also trim upstream dependency tests.")
    args = parser.parse_args()
    prepare_nltk()
    if args.vercel_build:
        trim_dependency_tests()
