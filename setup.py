from setuptools import setup

setup(name="gym_cap",
      version="1.0",
      author="DOs",
      license="MIT",
      packages=["gym_cap", "gym_cap.envs", "policy"],
      zip_safe=False,
      install_requires = ["gym", "pygame", "numpy"]
)
