from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

package_name = 'stereo_slam'

ext_modules = [
    Pybind11Extension(
        'stereo_slam_pybind',
        ['stereo_slam/stereo_slam_pybind/stereo_slam_pybind.cc'],
    )
]
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    ext_modules=ext_modules,
    install_requires=['setuptools', 'opencv-python', 'numpy', 'scipy', 'pyyaml', 'tqdm'],
    zip_safe=True,
    maintainer='junlinp',
    maintainer_email='junlinp@deepmirror.com',
    description='Python ROS 2 package example',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_slam = stereo_slam.my_node:main',
            'euroc_data_node = stereo_slam.euroc_data_node:main'
        ],
    },
)
