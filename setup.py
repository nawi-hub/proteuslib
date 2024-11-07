#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
"""
Project setup with setuptools
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from pathlib import Path

cwd = Path(__file__).parent
long_description = (cwd / "README.md").read_text()

SPECIAL_DEPENDENCIES_FOR_RELEASE = [
    "idaes-pse>=2.6.0,<2.7.0rc0",  # from PyPI
]

SPECIAL_DEPENDENCIES_FOR_PRERELEASE = [
    # update with a tag from the nawi-hub/idaes-pse
    # when a version of IDAES newer than the latest stable release from PyPI
    # will become needed for the watertap development
    "idaes-pse @ git+https://github.com/watertap-org/idaes-pse@2.7.0.dev0.watertap.24.10.10",
]

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="watertap",
    url="https://github.com/watertap-org/watertap",
    version="1.2.dev0",
    description="WaterTAP modeling library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NAWI team",
    license="BSD",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="water systems, chemical engineering, process modeling, filtration, desalination, nawi",
    # just include watertap and everything under it
    packages=find_packages(
        include=("watertap*",),
    ),
    python_requires=">=3.8",
    install_requires=[
        # primary requirements for unit and property models
        # maintainers: switch to SPECIAL_DEPENDENCIES_FOR_RELEASE when cutting a release of watertap
        *SPECIAL_DEPENDENCIES_FOR_PRERELEASE,
        "pyomo>=6.6.1",
        "flexparser != 0.4",  # IDAES/idaes-pse#1524
        "pyyaml",  # watertap.core.wt_database
        # for parameter_sweep
        "parameter-sweep>=0.1.dev5",
        # for watertap.ui.api_model (though may be generally useful)
        "pydantic >= 2, <3",
        "numpy",
        "importlib-resources",
    ],
    extras_require={
        "testing": [
            "pytest",
            "pandas",
            # treebeardtech/nbmake#121
            "nbmake != 1.5.1",
            "nbconvert",
        ],
        "notebooks": [
            "jupyter",
            "ipykernel",
        ],
        "oli_api": [
            "requests",
            "cryptography",  # for encrypting OLI credentials
        ],
        "dev": [
            "nbsphinx",  # jupyter notebook support for sphinx
            "jinja2<3.1.0",  # see watertap-org/watertap#449
            "Sphinx==7.1.*",  # docs
            "sphinx_rtd_theme",  # docs
            "urllib3 < 2",  # see watertap-org/watertap#1021,
            # other requirements
            "linkify-it-py",
            "black",  # code formatting
            # other requirements
            "pytest",  # test framework
            "pytest-cov",  # code coverage
            # treebeardtech/nbmake#121
            "nbmake != 1.5.1",
        ],
    },
    package_data={  # Optional
        "": [
            "*.json",
            "*.yaml",
            "*.yml",
            "*.csv",
            "*.png",
        ],
        "watertap.tools.oli_api.tests": [
            "test.dbs",
        ],
    },
    entry_points={
        "watertap.flowsheets": [
            "nf = watertap.flowsheets.nf_dspmde.nf_ui",
            "bsm2 = watertap.flowsheets.full_water_resource_recovery_facility.BSM2_ui",
            "bsm2_P_extension = watertap.flowsheets.full_water_resource_recovery_facility.BSM2_P_extension_ui",
            "dye_desalination = watertap.flowsheets.dye_desalination.dye_desalination_ui",
            "mvc = watertap.flowsheets.mvc.mvc_single_stage_ui",
            "RO = watertap.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery_ui",
            "OARO = watertap.flowsheets.oaro.oaro_multi_ui",
            "GAC = watertap.flowsheets.gac.gac_ui",
            "ED_conc_recirc = watertap.flowsheets.electrodialysis.electrodialysis_1stack_conc_recirc_ui",
            "LSRRO = watertap.flowsheets.lsrro.lsrro_ui",
            "generic desal train = watertap.flowsheets.generic_desalination_train.generic_train_ui",
        ],
    },
)
