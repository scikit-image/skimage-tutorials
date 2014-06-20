import skimage
test_failed = bool(skimage.test_verbose())
doctest_failed = bool(skimage.doctest_verbose())
if test_failed or doctest_failed:
    print("There were errors running scikit-image tests. "
          "See test output for more details.")
skimver = int(skimage.__version__.split('.')[1].rstrip('dev'))
if skimver < 10:
    print("skimage version 0.10 or higher required, you have 0.%i." % skimver)

