"""
This is the code for directory parsing
in order to make label files for 
UCF-CRIME datastet
"""

import glob
import os
#print(glob.glob("E:\UCF-crime dataset\Normal_Videos_event\*.avi"))
a = [name for name in os.listdir(".") if name.endswith(".mp4")]
a.sort()
print(a)
with open('1-listfile.txt', 'w') as filehandle:
    for listitem in a:
        filehandle.write('Abnormal/%s\n' % listitem)
print('done')
