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

`pip install skunkwork`

'''




from setuptools import setup
import os
import re
def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)


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

do_setup = False
if do_setup:

    setup(
        name=project_name,
        version=get_property('__version__', project_name),
        description="Python support functions",
        url="https://github.com/basameera/SkunkWork",
        author=get_property('__author__', project_name),
        author_email=get_property('__author_email__', project_name),
        packages=[project_name],
        # install_requires=dependancy_packages
    )
