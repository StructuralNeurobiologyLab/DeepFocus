from setuptools import find_packages, setup

setup(
    name='DeepFocus',
    version='1.0',
    description='Deep learning based focus and stigmation correction.',
    url='',
    download_url='',
    author='Philipp Schubert et al.',
    author_email='pschubert@neuro.mpg.de',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Connectomics :: Analysis Tools',
        'License :: OSI Approved :: '  # GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.7',
    ],
    platforms=["Linux", "Windows"],
    keywords='electronmicroscopy autofocus autostigmation',
    packages=find_packages(),
    python_requires='>=3.7, <4',
)
