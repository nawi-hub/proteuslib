###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################
# WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
#
# This module is a work in progress. Do not use it for real work right now.
#
# WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING

"""
JSON schema embedded as variables for:
  - component
  - reaction
"""

_parameter_def = {
    "type": "array",
    "description": "List of parameter values",
    "items": {
        "type": "object",
        "description": "Value, units, etc. for a parameter",
        "properties": {
            "v": {"description": "value", "type": "number"},
            "u": {"description": "units", "type": "string"},
            "i": {
                "oneOf": [
                    {"type": "string", "description": "string index"},
                    {"type": "number", "description": "numeric index"},
                ]
            },
        },
        "required": ["v", "u"],
    },
}

schemas = {
    "component": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://nawi-hub.github.com/electrolytedb/component",
        "type": "object",
        "description": "A chemical species that is a component in a reaction",
        "properties": {
            "name": {
                "description": "The chemical name of the component",
                "examples": ["HPO4 2-", "H3PO4", "NH4 +"],
                "type": "string",
            },
            "valid_phase_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Valid phase types should start with 'PT.' and then match "
                    "attributes in idaes.core.phases.PhaseType",
                    "examples": [["PT.liquidPhase"]],
                },
            },
            "phase_equilibrium_form": {
                "type": "object",
                "properties": {"Vap": {"type": "string"}, "Liq": {"type": "string"}},
            },
            "parameter_data": {
                "type": "object",
                "properties": {
                    "mw": {"$ref": "#/definitions/parameter"},
                    "pressure_crit": {"$ref": "#/definitions/parameter"},
                    "temperature_crit": {"$ref": "#/definitions/parameter"},
                },
                "patternProperties": {
                    "^.*_coeff$": {"$ref": "#/definitions/parameter"},
                    "^.*_ref$": {"$ref": "#/definitions/parameter"},
                },
                "additionalProperties": False,
            },
        },
        "required": ["name", "parameter_data"],
        "patternProperties": {"_comp": {"type": "string"}},
        "additionalProperties": False,
        "definitions": {"parameter": _parameter_def},
    },
    "reaction": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://nawi-hub.github.com/electrolytedb/reaction",
        "description": "The stoichiometry and properties of a reaction",
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["equilibrium"],
                "description": "Type of reaction",
            },
            "name": {"type": "string", "description": "Name of reaction"},
            "stoichiometry": {
                "type": "object",
                "description": "Moles for the given species in the reaction. Negative for LHS, positive for RHS. "
                "Grouped by phase.",
                "properties": {
                    "Liq": {"$ref": "#/definitions/stoichiometry"},
                    "Vap": {"$ref": "#/definitions/stoichiometry"},
                },
                "additionalProperties": False,
            },
            "heat_of_reaction": {"type": "string"},
            "equilibrium_constant": {"type": "string"},
            "equilibrium_form": {"type": "string"},
            "concentration_form": {"type": "string"},
            "parameter_data": {
                "type": "object",
                "patternProperties": {"_ref": {"$ref": "#/definitions/parameter"}},
                "additionalProperties": False,
            },
        },
        "required": ["name", "parameter_data", "type"],
        "definitions": {
            "parameter": _parameter_def,
            "stoichiometry": {
                "description": "One part of the stoichiometry",
                "examples": ['{"NH4 +": -1}'],
                "type": "object",
                "patternProperties": {
                    "^[A-Z].*$": {
                        "type": "number",
                        "description": "Moles for the given species in the reaction. Negative for LHS, positive for RHS",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
}
