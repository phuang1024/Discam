from setuptools import setup, find_packages

from discam.utils.constants import VERSION


with open("README.md", "r") as f:
    long_des = f.read()

with open("requirements.txt", "r") as f:
    reqs = f.read().strip().split("\n")

setup(
    name="discam",
    version=VERSION,
    author="Patrick Huang",
    url="https://github.com/phuang1024/Discam",
    description="Discam is a set of computer vision tools for automated filming of Ultimate Frisbee games.",
    long_description=long_des,
    long_description_content_type="text/markdown",

    install_requires=reqs,
    packages=["discam", "discam.cv", "discam.post", "discam.utils"],
    entry_points={
        "console_scripts": [
            "discpost = discam.post_main:main",
            "discmask = discam.utils.field_mask:main",
        ]
    },
)
