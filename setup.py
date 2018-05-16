from setuptools import setup, find_packages

setup(
    name= 'mabandit',
    packages = find_packages(),
    version= '1.3',
    description= "A library for mutliarm bandit problem",
    author= 'Pankayaraj',
    author_email = 'pankayaraj1995@gmail.com',
    url = 'https://github.com/punk95/Multi-Arm-Bandit-Library',

    keywords = ['mabandit', 'pypi', 'package'],
    install_requires = ["numpy", "scipy","tensorflow >= 1.4"],
    classifiers= [
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]


)
