###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################
"""
    This file demonstrates the basics of working with and using the electrolyte
    database (EDB).

    (1) Before we can start, you must install MongoDB (which is installed separately)

        [See more information on the ReadTheDocs under 'Getting Started --> Installing WaterTAP']

    (2) After installing MongoDB, you will need to 'load' the database using the
        command line function 'edb load -b'. This will load the default database
        that WaterTAP is bundled with.

        [NOTE: If you need to 'reload' the database, simply use the command 'edb drop -d electrolytedb'
        in the command line. The database on MongoDB is named "electrolytedb"]

        [NOTE 2: You can invoke the command line utility with the "help" keyword to
        get more information on funtionality. Command: 'edb --help' or 'edb [arg] --help']

    (3) To use EDB in python, start by importing the interface class object 'ElectrolyteDB'

    (4) Invoke the 'ElectrolyteDB' object to connect to the database

        [# WARNING: A feature of 'ElectrolyteDB' appears to be missing from WaterTAP main
        that is preventing us from following the tutorial verbatim.]

    (5) Grab a 'base' for a configuration dictionary, and place it into a class object

        [# WARNING: There are a number of ways to do this. Behavior is inconsistent with
        some of the tutorials currently.]

"""

# ========================== (3) ================================
# Import ElectrolyteDB object
from watertap.edb import ElectrolyteDB

# ========================== (4) ================================
# By default, invoking the 'ElectrolyteDB' object (with no args)
#   will attempt to connect to the local host database. You can
#   check the connection by calling the 'can_connect' function
#   and passing the 'host' and 'port' as args. If no 'host' or
#   'port' are given, then it uses the defaults.
def connect_to_edb():
    print("connecting to " + str(ElectrolyteDB.DEFAULT_URL))
    db = ElectrolyteDB()
    connected = db.can_connect()

    # Supposedly, we should be able to pass a check_connection arg to the
    #   initialization of the 'ElectrolyteDB' object. However, that does
    #   not appear to be a feature that exists in WaterTAP main.
    try:
        db2 = ElectrolyteDB("mongodb://some.other.host", check_connection=False)
        print("Everything is good!")
    except:
        print("If you are here, that means that the EDB code is not up-to-date")

    return (db, connected)

# ========================== (5) ================================
# All configuration files used in WaterTAP for electrolyte chemistry
#   require a 'base' dictionary to start. For example, we need to
#   create a 'thermo_config' dictionary to pass to the GenericProperty
#   package in IDAES. That 'thermo_config' file will always have a
#   few specific items in common with most other configuration files.
#   Thus, this operation will populate a class object that holds the
#   data assocated with that 'base' dictionary.
#
# In the EDB, there are several different 'base' structures to start
#   from. In this example, we will build from the default 'thermo'
#   configuration base.
#
#   NOTE: There is a difference between the 'get_base' function and
#       the 'get_one_base' function. The 'get_base' function will
#       return a 'Result' object that contains an iterator for each
#       base requested. This means that in order to access the config
#       you just created, you would need to iterate through that
#       result object and look at the 'idaes_config' for each obj
#       in the 'result'. Alternatively, you can use the 'get_one_base'
#       function, which will directly return a single object that
#       can be directly queried for the 'idaes_config'.
def grab_base_thermo_config(db):

    # Get the base and place into a result object that
    #   needs to be iterated through to access data at
    #   each object held in the result
    res_it_obj = db.get_base("thermo")
    for obj in res_it_obj:
        obj.idaes_config  # This is where you access the data

    # Get the base and place into a singular object
    #   with no need to iterate through
    res_obj = db.get_one_base("thermo")
    res_obj.idaes_config  # This gives more direct access to the information
    return res_obj

if __name__ == "__main__":
    (db, connected) = connect_to_edb()
    if (connected == False):
        print("Failed to connect")
    else:
        print("Now connected")

    res_obj = grab_base_thermo_config(db)
    print(res_obj.idaes_config)
