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
        "langchain==0.3.7",
        "langchain_community==0.3.5",
        "nltk==3.8.1",
        "openai==1.54.3",
        "python-dotenv==1.0.1",
        "setuptools==59.6.0",
        "streamlit==1.40.0",
        "tiktoken",
    ],
)
