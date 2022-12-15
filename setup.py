from setuptools import setup,find_packages


with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()


setup(
    name="axel",
    version="0.0.1",
    description="",
    long_description=readme,
    author="https://github.com/rusenburn",
    url="https://github.com/rusenburn",
    license=license,
    packages=find_packages(exclude=('tests','docs'))
)