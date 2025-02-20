#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 示例
from setuptools import setup, find_packages

setup(
    name="neuro-lab",  # 包名称
    version="0.1.0",  # 版本号
    author="DanFienne",
    author_email="danfeng19920912@gmail.com",
    description="A deep learning and reinforcement learning toolkit",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuroai",  # 你的 GitHub 主页（可选）
    packages=find_packages(),  # 自动查找所有 Python 包
    install_requires=[
        "numpy",
        "torch",
    ],  # 依赖库
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # 需要的 Python 版本
)