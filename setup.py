from setuptools import setup, find_packages

# this is to work around a flash-attn install bug: https://github.com/Dao-AILab/flash-attention/issues/453
from pip._internal import main as pip_main
pip_main(['install', 'packaging', 'wheel', 'torch==2.0.1'])

# normal install
setup(
    name="llava",
    version="1.1.3",
    description="Towards GPT-4 like large language and visual assistant.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="",  # Add the author's name if applicable
    author_email="",  # Add the author's email if applicable
    url="https://llava-vl.github.io",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    packages=find_packages(
        exclude=["checkpoints*", "datasets*", "hf*", "assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]
    ),
    install_requires=[
	    "torch==2.0.1", 
        "torchvision==0.15.2",
        "transformers==4.35.0", 
        "tokenizers==0.14.1", 
        "sentencepiece==0.1.99", 
        "shortuuid",
        "accelerate==0.24.0", 
        "peft==0.6.1", 
        "bitsandbytes==0.41.2.post2",
        "pydantic<2,>=1", 
        "markdown2[all]", 
        "numpy", 
        "scikit-learn==1.2.2",
        "gradio==3.35.2", 
        "gradio_client==0.2.9",
        "requests", 
        "httpx==0.24.0", 
        "uvicorn", 
        "fastapi",
        "einops==0.6.1", 
        "einops-exts==0.0.4",
        "flash-attn==2.3.3",
        "open_clip_torch @ git+https://github.com/rmst/open-clip.git@01f8200c02c79f582d1189eda88e7459ac1994fe",
        "timm @ git+https://github.com/rmst/pytorch-image-models.git@2eea97d64bbe2abc9c75b19380d7fbdb22c872f2"
    ],
    extras_require={
        "train": ["deepspeed==0.12.2", "ninja", "wandb"]
    },
    project_urls={
        "Bug Tracker": "https://github.com/haotian-liu/LLaVA/issues",
    },
    include_package_data=True,
    zip_safe=False
)
