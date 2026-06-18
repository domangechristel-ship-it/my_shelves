"""Setup main my_shelves package."""

from setuptools import find_packages
from setuptools import setup


with open("requirements.txt", encoding="utf-8") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if x.strip() and not x.startswith("git+") and not x.startswith("--")]


setup(name='my_shelves',
      version="0.0.4",
      description="MyShelves Books recommandations.",
      license="MIT",
      author="Le Wagon",
      author_email="contact@lewagon.org",
      #url="https://github.com/lewagon/taxi-fare",
      install_requires=requirements,
      packages=find_packages('src'),
      package_dir={'': 'src'},
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'process-data=my_shelves.prepare.process:main',
          ],
      })
