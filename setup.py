from pathlib import Path
import os
from setuptools import setup, find_packages
import glob

REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"


with open(str(REQUIREMENTS_PATH), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()


def dump_configs_to_cache():
    config_files = [
        (f'{os.environ["HOME"]}/.cache/lexsubgen/' + str(Path(file).parent), file)
        for file in list(glob.glob("configs/**/*.jsonnet", recursive=True))
    ]
    for file in config_files:
        dest_dir, src_path = file
        dest_dir = Path(dest_dir)
        src_path = Path(src_path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(src_path, "r") as src_fp, open(
            dest_dir / src_path.name, "w"
        ) as dst_fp:
            dst_fp.write(src_fp.read())


setup(
    name="LexSubGen",
    version="0.0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    # ['lexsubgen', 'lexsubgen.utils', 'lexsubgen.metrics', 'lexsubgen.analyzer', 'lexsubgen.analyzer.backend',
    #           'lexsubgen.datasets', 'lexsubgen.generators', 'lexsubgen.generators.pre_processing',
    #           'lexsubgen.generators.post_processing', 'lexsubgen.generators.prob_estimators',
    #           'lexsubgen.generators.prob_estimators.baseline', 'lexsubgen.evaluations', 'lexsubgen.applications',
    #           'lexsubgen.clusterizers'],
    url="",
    license="Apache",
    author="Samsung Electronics",
    author_email="",
    description="Framework for generating lexical substitutes",
    long_description=open("README.md").read(),
    keywords="lexsubgen NLP deep learning lexical substitution",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6.1",
    entry_points={
        "console_scripts": [
            "lexsubgen=lexsubgen.runner:main",
            "lexsubgen-app=lexsubgen.analyzer.backend.app:main",
        ]
    },
)
dump_configs_to_cache()
