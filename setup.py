from setuptools import setup

setup(
    name = "tensorsem",
    version = "1.0",
    description = "SEM as a pytorch module",
    url = "http://github.com/vankesteren/tensorsem",
    author = "Erik-Jan van Kesteren",
    author_email = "e.vankesteren1@uu.nl",
    license = "GPL3",
    packages = ["tensorsem"],
    install_requires = ["torch"],
    zip_safe = False,
    scripts = ["bin/tensorsem_cmd"],
    include_package_data = True
)