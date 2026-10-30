from setuptools import find_packages, setup

setup(
    name='lite3-supermimic',
    version='0.1.0',
    description='Lite3 model-based reinforcement learning with SuperMimic in mjlab',
    license='MIT AND MPL-2.0',
    packages=find_packages(include=['lite3_supermimic', 'lite3_supermimic.*']),
    python_requires='>=3.10',
    install_requires=['mjlab==1.6.0', 'torch', 'matplotlib', 'mediapy', 'pillow', 'tensorboard'],
    extras_require={'dev': ['pytest']},
    package_data={'lite3_supermimic': ['LICENSE_supermimic', 'robot/lite3.xml',
                                    'robot/LICENSE_lite3', 'robot/meshes/*.STL']},
)
