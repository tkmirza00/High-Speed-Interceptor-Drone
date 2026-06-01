from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mmwave_drone_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'drone_detector = mmwave_drone_detector.ros_publisher:main',
            'radar_filter_node = mmwave_drone_detector.radar_filter_node:main',
            'radar_range_markers = mmwave_drone_detector.radar_range_markers:main',
            
        ],
    },
)
