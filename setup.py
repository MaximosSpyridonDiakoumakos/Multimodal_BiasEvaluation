from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-bias-evaluation",
    version="1.0.0",
    author="Maximos Spyridon Diakoumakos",
    author_email="it2021027@hua.gr",
    description="A comprehensive toolkit for evaluating bias in multimodal AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaximosSpyridonDiakoumakos/Multimodal_BiasEvaluation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-bias-eval=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="bias evaluation, multimodal, AI, text-to-image, image-to-text, fairness",
    project_urls={
        "Bug Reports": "https://github.com/MaximosSpyridonDiakoumakos/Multimodal_BiasEvaluation/issues",
        "Source": "https://github.com/MaximosSpyridonDiakoumakos/Multimodal_BiasEvaluation",
        "Documentation": "https://github.com/MaximosSpyridonDiakoumakos/Multimodal_BiasEvaluation#readme",
    },
) 