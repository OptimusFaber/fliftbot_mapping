"""Set up the platform-independent Fliftbot mapping package."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'flifltbot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src', exclude=['test']),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config', 'cartographer'),
            [path for path in glob('config/cartographer/*')
             if os.path.isfile(path)]),
        (os.path.join('share', package_name, 'config', 'slam_toolbox'),
            [path for path in glob('config/slam_toolbox/*')
             if os.path.isfile(path)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OptimusFaber',
    maintainer_email='OptimusFaber@users.noreply.github.com',
    description='Platform-independent offline LiDAR mapping for Fliftbot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'map_build = flifltbot.common.cli.map_build:main',
            'map_snapshot = flifltbot.common.cli.map_snapshot:main',
            'map_check_inputs = flifltbot.common.cli.map_check_inputs:main',
            'map_eval_accuracy = flifltbot.common.cli.map_eval_accuracy:main',
            'map_export_real_gt = '
            'flifltbot.common.cli.map_export_real_gt:main',
            'map_factor_diag = flifltbot.common.cli.map_factor_diag:main',
            'npz_cartographer_replay = '
            'flifltbot.common.runtime.cartographer_replay:main',
        ],
    },
)
