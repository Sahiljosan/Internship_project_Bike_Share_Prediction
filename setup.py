from setuptools import find_packages, setup
from typing import List

requirement_file_name = "requirements.txt"
REMOVE_PACKAGE = "-e ."


def get_requirements()->List[str]:
    with open(requirement_file_name) as requirement_file:
        requirement_list = requirement_file.readline()
    requirement_list = [requirement_name.replace("\n","") for requirement_name in requirement_list]

    if REMOVE_PACKAGE in requirement_list:
        requirement_list.remove(REMOVE_PACKAGE)
    return requirement_list


setup(name='Bike_Share_Prediction',
      version='0.0.1',
      description='Bike_Share_Prediction_End_to_End Project',
      author='Sahil Josan',
      author_email='sahiljosan50@gmail.com',
      # url='https://www.python.org/sigs/distutils-sig/',
      packages = find_packages(),
      install_requires = get_requirements()
     )
