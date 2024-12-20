from setuptools import find_packages, setup
import glob

package_name = 'franka_hri_sorting'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.*')),
        ('share/' + package_name + '/config', glob.glob('config/*.rviz')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='scferro',
    maintainer_email='stephencferro@gmail.com',
    description='A package for managing human interactions ',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'blocks = franka_hri_sorting.blocks:main',
            'human_input = franka_hri_sorting.human_input:main',
            'human_interaction = franka_hri_sorting.human_interaction:main',
            'network_node = franka_hri_sorting.network_node:main',
            'network_training = franka_hri_sorting.network_training:main',
        ],
    },
    py_modules=[
        'franka_hri_sorting/network',  
    ],
)
