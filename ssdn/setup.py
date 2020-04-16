from setuptools import setup, find_packages

exec(open("ssdn/version.py").read())

setup(
    name="ssdn",
    version=__version__,  # noqa
    packages=find_packages(),
    install_requires=["nptyping", "h5py"],
)
