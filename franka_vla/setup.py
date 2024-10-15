from setuptools import find_packages, setup
import glob

package_name = 'franka_vla'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.*')),
    ],
    install_requires=['setuptools', 'opencv-python-headless'],
    zip_safe=True,
    maintainer='scferro',
    maintainer_email='stephencferro@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_publisher = franka_vla.webcam_publisher:main',
            'vla_node = franka_vla.vla_node:main',
        ],
    },
)