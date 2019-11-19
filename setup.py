"""SkunkWork setup.py
https://python-packaging.readthedocs.io/en/latest/index.html

Full Package Structure - https://python-packaging.readthedocs.io/en/latest/everything.html

"""
'''
1. Install custom package

`pip install .` - install current version
`pip install -e .` - install current verison, BUT FUTURE UPDATES WILL BE IMMEDATELY REFLECTED.

2. Test by `import SkunkWork`

3. Uninstall script

`pip uninstall skunkwork`

'''




from setuptools import setup
import os
import re

project_name = 'skunkwork'

def read_req(project_name):
    req_list = []
    fname = 'requirements.txt'
    if os.path.isfile(fname):
        with open(fname) as f:
            content = f.readlines()
            for line in content:
                req_list.append(line.split('==')[0])
    else:
        raise AttributeError(
            'No \'requirements.txt\' file. Please run \'bash pipreqs.sh\' to generate it.')
    return req_list


dependancy_packages = read_req(project_name)

do_setup = True
if do_setup:

    setup(
        name=project_name,
        version='0.1.3.191116',
        description="Python support functions",
        url="https://github.com/basameera/SkunkWork",
        author='Sameera Sandaruwan',
        author_email='basameera@pm.me',
        packages=[project_name],
        # install_requires=dependancy_packages
    )
