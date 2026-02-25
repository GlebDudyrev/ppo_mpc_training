import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_training'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='g_dudyrev',
    maintainer_email='g_dudyrev@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'target_creator = tb3_training.target_creator:main',
            'check_env = tb3_training.check_env:main',
            'tb3_train = tb3_training.tb3_train:main',
            'tb3_train_mpc = tb3_training.tb3_train_mpc:main',
        ],
    },
)
