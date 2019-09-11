import sys
from distutils.version import LooseVersion

if sys.version_info.major < 3 or sys.version_info.minor < 5:
    print('[!] You are running an old version of Python. '
          'This tutorial requires Python 3.5.')

    sys.exit(1)

with open('requirements.txt') as f:
    reqs = f.readlines()

reqs = [(pkg, ver) for (pkg, _, ver) in
        (req.split() for req in reqs if req.strip())]

pkg_names = {
    'jupyter-notebook': 'notebook',
    'scikit-image': 'skimage',
}

for (pkg, version_wanted) in reqs:
    module_name = pkg_names.get(pkg, pkg)
    try:
        m = __import__(module_name)
        status = '✓'
    except ImportError as e:
        m = None
        if (pkg != 'numpy' and 'numpy' in str(e)):
            status = '?'
            version_installed = 'Needs NumPy'
        else:
            version_installed = 'Not installed'
            status = 'X'

    if m is not None:
        try:
            version_installed = m.__version__
        except AttributeError:  # specific for ITK version
            version_installed = m.Version.GetITKVersion()

        if LooseVersion(version_wanted) > LooseVersion(version_installed):
            status = 'X'
    print('[{}] {:<10} {}'.format(
        status, pkg.ljust(16), version_installed)
        )
