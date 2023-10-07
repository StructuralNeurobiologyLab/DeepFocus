from setuptools import setup

setup(
    name='deepfocus',
    version='1.0',
    description='Deep learning based focus and astigmatism correction for electron microscopy.',
    url='',
    download_url='',
    author='Philipp Schubert',
    author_email='phil.jo.schubert@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Connectomics :: Analysis Tools',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License'
    ],
    platforms=["Linux", "Windows"],
    keywords='electronmicroscopy autofocus autostigmation',
    packages=['deepfocus'],
    python_requires='>=3.7, <4',
)
