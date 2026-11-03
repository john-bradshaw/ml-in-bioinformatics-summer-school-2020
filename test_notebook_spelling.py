import pathlib
import re

REPO_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOKS = ['ML_for_Molecules.ipynb', 'ML_for_Molecules_solutions.ipynb']
# Variable names such as paracetemol_str are code identifiers and are left alone.
MISSPELLING = re.compile(r'paracetemol(?!_)', re.IGNORECASE)


def test_no_paracetemol_misspelling_in_notebooks():
    for name in NOTEBOOKS:
        text = (REPO_DIR / name).read_text(encoding='utf-8')
        matches = MISSPELLING.findall(text)
        assert not matches, f"{name} has {len(matches)} misspelling(s) of 'paracetamol'"
