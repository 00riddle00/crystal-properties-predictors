import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements: list = [
    # Use either environment.yml (conda) or
    # requirements/{requirement_file}.txt (pip).
]

setup(
    name="crystal_property_predictor",
    version="0.0.1",
    url="https://gitlab.com/the-study-of-crystal-structures-using-machine-learning/crystal-properties-predictors",  # noqa
    author="Tomas Giedraitis",
    author_email="tomasgiedraitis@gmail.com",
    description="Crystal properties prediction from structure.",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "crystal_property_predictor=crystal_property_predictor.cli:cli"
        ]
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
    ],
)
