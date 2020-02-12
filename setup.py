from setuptools import setup

setup(name="gym_cap",
      version="1.0.0",
      author="DOs, skim449",
      license="MIT",
      url="https://https://github.com/raide-project/ctf_public",
      packages=["gym_cap", "gym_cap.envs"],
      zip_safe=False,
      install_requires = [
          "gym>=0.16.0",
          "pygame>=1.3.2",
          "numpy>=1.18.1"
          ]
)
