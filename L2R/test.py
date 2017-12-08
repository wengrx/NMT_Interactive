import ctypes
ll = ctypes.cdll.LoadLibrary
lib = ll('./alignlib/libalign.so')
lib.test()
lib.loadalldata()
lib.startalign('transmt03_1', 'ref2', '/data/align/mt03_ref_1/')
# lib.startalign('transmt03', 'ref2', '/data/align/mt03_ref_0/')


# import os
#
# if os.path.exists('./test'):
#     print 'True'
# else:
#     os.makedirs('./test')