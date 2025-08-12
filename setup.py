from setuptools import find_packages,setup
from typing import List
# setup.py

HYPEN = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirements
    as per the file path provided.
    '''
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN in requirements:
            requirements.remove(HYPEN)  
    return requirements

setup(
name='SMS-classifier',
version='0.1',
author='BinudaDewha',
author_email='binudab4@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),

)