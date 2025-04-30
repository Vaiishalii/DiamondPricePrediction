from setuptools import find_packages,setup
from typing import List

hypen_e_dot='-e.'

def get_requirements(file_path:str)->list[str]:
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
    requirements=[r.replace("\n","") for r in requirements]

    if hypen_e_dot in requirements:
        requirements.remove(hypen_e_dot)

    return requirements


setup(
    name='DiamondPricePredictionMLProject',
    version='0.0.1',
    author='Vaishali',
    author_email='vaiishaliisharma31@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)