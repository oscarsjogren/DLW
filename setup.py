from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='dlw',
      version='0.1',
      description='DLW pricing model',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='DLW social cost of carbon',
      url='http://github.com/dlw/dlw',
      author='Robert Litterman',
      author_email='asads@gmail.com',
      license='MIT',
      packages=['dlw'],
      install_requires=['numpy',],
      include_package_data=True,
      zip_safe=False)

