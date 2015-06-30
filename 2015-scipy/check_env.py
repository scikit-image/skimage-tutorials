# Check skimage along with major dependencies: numpy, scipy, matplotlib
import skimage
test_failed = bool(skimage.test())
if test_failed:
    print("There were errors running scikit-image tests. "
          "See test output for more details.")
skimver = int(skimage.__version__.split('.')[1].rstrip('dev'))
if skimver < 11:
    print("ERROR: skimage version 0.11 or higher required. You have 0.%i." %
          skimver)

# check matplotlib
import matplotlib as mpl
mplver_str = mpl.__version__
mplver = mplver_str.split('.')
major, minor = map(int, mplver[:2])
if major < 1 or (major >= 1 and minor < 1):
    print("ERROR: matplotlib version 1.1 or higher required. You have %s." %
          mplver_str)

# Check IPython
import IPython as IP
ipver_str = IP.__version__
ipver = int(ipver_str.split('.')[0])
if ipver < 3:
    print("ERROR: IPython version 3.0 or higher required. You have %s." %
          ipver_str)
