from setuptools import setup, find_packages

import versioneer

with open('README.md', "r") as file:
    read_me = file.read()

setup(
    name='mmproteo',
    version=versioneer.get_version(),
    description='Mirko meets Proteomics: A PRIDE Archive downloader for deep '
                'learning-based DeNovo Sequencing',
    long_description=read_me,
    long_description_content_type="text/markdown",
    url='https://github.com/Miroka96/mmproteo',
    author='Mirko Krause',
    author_email='krause@codebase.one',
    license='GPLv3+',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        "console_scripts": ['mmproteo = mmproteo.mmproteo:main']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: Console",
        "License :: OSI Approved :: "
        "GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "requests",
        "pandas",
        "pyteomics",
        "wget",
        "pyarrow",
        "lxml",
        "numpy"
    ],
    cmdclass=versioneer.get_cmdclass(),
)
