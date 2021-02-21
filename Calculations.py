from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import numpy as np

def density(xyxy):

    return

def road(dict,frame,resx,resy):
    xarray = []
    yarray = []

    box = {}
    for x in range(1, 10):
        for y in range(1, 10):
            key = f"{x, y}"
            box[key] = []
    ymax = resx
    xmax = resy

    dontcountarray = []
    for entry in dict:
        sumarray = []
        car = dict[entry]
        for frame in car:
            value = car[frame]
            sumarray.append(value[0] + value[1] + value[2] + value[3])
            range2 = max(sumarray) - min(sumarray)
        if range2 < 100 * ((1080*720)/(ymax*xmax)):
            dontcountarray.append(entry)

    carsarray  = []
    for entry in dict:
        car = dict[entry]
        if entry not in dontcountarray:
            for frame in car:
                value = car[frame]
                midx = value[2] - (value[2] - value[0])
                midy = value[3] - (value[3] - value[1])

                xarray.append(midx)
                yarray.append(midy)
                for key in box:
                    x = int(key[1])
                    y = int(key[4])

                    if midx != 0 and midy != 0:
                        if midx > (x - 1) * (1/9) * xmax and midx < x * (1/9) * xmax and midy > (y - 1) * (1/9) * ymax and midy < y * (1/9) * ymax:
                            box[key].append(1)
    roadi = []
    coords = []
    for key in box:
        x = int(key[1]) * (1/9) * xmax
        y = int(key[4]) * (1/9) * ymax
        width = xmax / 9
        height = ymax / 9

        coord = [int(x-(width/2)),int(y+(height/2)),int(x+(width/2)),int(y-(height/2))]

        cord1 = np.array([int(x-(width/2)),int(y+(height/2))])
        cord2 = np.array([int(x+(width/2)),int(y-(height/2))])

        count = len(box[key])
        if count > 0:
            roadi.append(coord)
            coords.append(cord1)
            coords.append(cord2)
    nparray = np.array(coords)
    newcoord = []
    try:
        results = ConvexHull(nparray)
        for simplex in results.simplices:
            print(nparray[simplex,0],nparray[simplex,1])
            newcoord.append([nparray[simplex, 0][0], nparray[simplex, 1][0]])
            newcoord.append([nparray[simplex, 0][1], nparray[simplex, 1][1]])

    except:
        print("Not working")

    return roadi, newcoord

def velocity(outputs):
    print("velocity")
    return

def vehcileTracker(outputs):
    print("vehicleTracker")
    return


def correlation(carsdict, xyxy, framenumber, resx, resy, fps):
    diffx = (xyxy[2]-xyxy[0])
    diffy = (xyxy[3]-xyxy[1])
    ratio = ((diffx+diffy)/2)/30
    threshold = 30*((1080*720)/(resx*resy))*(ratio)*(30/fps)
    iterlen = len(carsdict)
    validated = False
    carnum = 1
    for i in range(iterlen):
        one = str(framenumber-1)
        two = str(framenumber-2)
        three = str(framenumber-3)
        four = str(framenumber-4)
        five = str(framenumber-5)

        try:
            old1 = carsdict[str(i+1)][one]
        except:
            old1 = [1000000,1000000,100000,1000000]
        try:
            old2 = carsdict[str(i+1)][two]
        except:
            old2 = [1000000,1000000,100000,1000000]
        try:
            old3 = carsdict[str(i+1)][three]
        except:
            old3 = [1000000,1000000,100000,1000000]
        try:
            old4 = carsdict[str(i + 1)][four]
        except:
            old4 = [1000000, 1000000, 100000, 1000000]
        try:
            old5 = carsdict[str(i + 1)][five]
        except:
            old5 = [1000000, 1000000, 100000, 1000000]
        if abs(old1[0]-xyxy[0]) <= threshold and abs(old1[1]-xyxy[1]) <= threshold and abs(old1[2]-xyxy[2]) <= threshold and abs(old1[3]-xyxy[3]) <= threshold:
            carsdict[str(i+1)][str(framenumber)] = xyxy
            validated = True
            carnum = i + 1
            break
        if abs(old2[0]-xyxy[0]) <= threshold and abs(old2[1]-xyxy[1]) <= threshold and abs(old2[2]-xyxy[2]) <= threshold and abs(old2[3]-xyxy[3]) <= threshold:
            carsdict[str(i+1)][str(framenumber)] = xyxy
            validated = True
            carnum = i + 1
            break
        if abs(old3[0]-xyxy[0]) <= threshold and abs(old3[1]-xyxy[1]) <= threshold and abs(old3[2]-xyxy[2]) <= threshold and abs(old3[3]-xyxy[3]) <= threshold:
            carsdict[str(i+1)][str(framenumber)] = xyxy
            validated = True
            carnum = i + 1
            break
        if abs(old4[0]-xyxy[0]) <= threshold and abs(old4[1]-xyxy[1]) <= threshold and abs(old4[2]-xyxy[2]) <= threshold and abs(old4[3]-xyxy[3]) <= threshold:
            carsdict[str(i+1)][str(framenumber)] = xyxy
            validated = True
            carnum = i + 1
            break
        if abs(old5[0]-xyxy[0]) <= threshold and abs(old5[1]-xyxy[1]) <= threshold and abs(old5[2]-xyxy[2]) <= threshold and abs(old5[3]-xyxy[3]) <= threshold:
            carsdict[str(i+1)][str(framenumber)] = xyxy
            validated = True
            carnum = i + 1
            break
    if validated == False:
        length = len(carsdict)
        carsdict[str(length+1)] = {str(framenumber):xyxy}
        carnum = length + 1
    return carsdict, carnum


