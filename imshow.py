#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import PIL.Image as Image

def inferShape(total_bytes:int):
    i,r = divmod(total_bytes, 4)
    if r:
        raise ValueError('Got %d bytes, not times of 4 (RGBA)' % total_bytes)
    j = int(math.sqrt(i))
    k,q = j,None
    for k in range(j,0,-1):
        q,r = divmod(i,k)
        if 0 == r:
            break
    print(k,q,4)
    return (k,q,4)

arr = np.fromfile('a.raw', dtype='uint8')
img = Image.fromarray(arr.reshape(inferShape(len(arr))))
img.save('a.png')
img.show()
