from setuptools import setup, find_packages

setup(
    name='weather_wizard',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "numpy>=1.19.0",
    ],
    entry_points={
        'console_scripts': [
            'weather_wizard=app.main:main',
        ],
    },
    python_requires='>=3.7',
    )