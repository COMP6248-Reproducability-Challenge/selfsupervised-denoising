from setuptools import setup, find_packages

exec(open("ssdn/version.py").read())

setup(
    name="ssdn",
    version=__version__,
    packages=find_packages(),
    install_requires=[],
)
