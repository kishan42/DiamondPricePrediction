from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements from the given file path.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="DiamondPricePredictor",
    version="0.0.1",
    author="Kishan",
    author_email="kishankachhadiya823@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
    )