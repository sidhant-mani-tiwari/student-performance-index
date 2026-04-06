from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filepath: str) -> List[str]:
    '''
    Docstring for get_requirements
    
    :param filepath: Description
    :type filepath: str
    :return: Description
    :rtype: List[str]
    This function will return a list of requirements
    '''
    requirements=[]
    with open(filepath) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="student-performance-index",
    version='0.0.1',
    author='Sidhant Mani Tiwari',
    author_email='sidhantmanitiwari@gmail.com',
    packages=find_packages(),
    intstall_requires=get_requirements('requirements.txt')
)