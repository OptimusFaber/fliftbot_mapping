"""Install the self-contained navigation runtime and reference sensor profiles."""
from pathlib import Path
from setuptools import find_packages, setup

PACKAGE = "fliftbot_lidar_nav"
data_files = [("share/ament_index/resource_index/packages", ["resource/" + PACKAGE]),
              ("share/" + PACKAGE, ["package.xml", "README.md", "LICENSE"])]
for directory in ("scripts", "config", "launch", "maps", "tests", "docs"):
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            data_files.append((str(Path("share") / PACKAGE / path.parent), [str(path)]))
setup(name=PACKAGE, version="0.1.0", packages=find_packages("src"),
      package_dir={"": "src"}, data_files=data_files,
      install_requires=["setuptools", "numpy", "scipy", "Pillow", "PyYAML"],
      tests_require=["pytest"], zip_safe=False, maintainer="OptimusFaber",
      maintainer_email="OptimusFaber@users.noreply.github.com", license="MIT",
      description="Fliftbot LiDAR localization on fliftbot_map_build maps",
      entry_points={"console_scripts": ["lidar_nav = fliftbot_lidar_nav.cli:main"]})
