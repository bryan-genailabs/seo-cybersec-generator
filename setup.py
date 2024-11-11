from setuptools import setup, find_packages

setup(
    name="cybersec_blog_generator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.32.0",
        "beautifulsoup4>=4.12.3",
        "requests>=2.31.0",
    ],
)
