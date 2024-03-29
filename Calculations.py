from scipy.spatial import ConvexHull
import numpy as np
from functools import reduce
import operator
import math
from shapely.geometry import Point, Polygon


def validcars(dict,resx,resy, framenumber):
    """
    :param dict: Car Dictionary Consisting of all of the Detections by Vehicle Number
    :param resx: Video Resolution x
    :param resy: Video Resolution Y
    :param framenumber: Current Frame Number
    :return: Returns an array of car that do not move (ie. are bad detections)
    """
    dontcountarray = []
    cararea = 0
    carnum = 0

    for entry in dict:
        sumarray = []
        car = dict[entry]
        for dictframe in car:
            value = car[dictframe]
            sumarray.append(value[0] + value[1] + value[2] + value[3])
            range2 = max(sumarray) - min(sumarray)
        if range2 < 100 * ((1080 * 720) / (resx * resy)):
            dontcountarray.append(entry)
        elif f"{framenumber}" in car:
            carnum += 1
            cararea += (value[2]-value[0])*(value[3]-value[1])

    return dontcountarray

def flowrate(dict,fps,dontcount,start,end):

    """
    :param dict: Car Dictionary Consisting of all of the Detections by Vehicle Number
    :param fps: Frames per second of video
    :param dontcount: Cars that are bad detections
    :param start: Starting frame for caluclation
    :param end: End frame for calculation
    :return: Returns the flow in and flow out over given time frame
    """

    into = 0
    out = 0
    for entry in dict:
        car = dict[entry]
        if entry not in dontcount:
            firstframe = int(list(car.keys())[0])
            lastframe = int(list(car.keys())[-1])
            if firstframe > start:
                into += 1
            if lastframe < end:
                out += 1

    try:
        flowin = (into * fps * 60)/end
        flowout = (out * fps* 60)/end
    except:
        flowin = 0
        flowout = 0

    return flowin,flowout

def carsonroad(detections,ordered):

    """
    :param detections: Detections for the current frame
    :param ordered: Polygon for the road
    :return: Number of cars, area of road, coordinates for detections in the area of the road
    """
    converted = []
    for point in ordered:
        converted.append((point[0],point[1]))
    poly = Polygon(converted)
    gooddetections = []
    numcars = 0
    area = 0
    for point in detections:
        x = point[2]-((point[2] - point[0])/2)
        y = point[3]-((point[3] - point[1])/2)
        p1 = Point(x,y)
        if p1.within(poly):
            numcars += 1
            area += (point[2]-point[0])*(point[3]-point[1])
            gooddetections.append([point[0],point[1],point[2],point[3]])

    return numcars, area, gooddetections

def road(dict,resx,resy,dontcountarray):
    """
    :param dict: Car Dictionary Consisting of all of the Detections by Vehicle Number
    :param resx: Video Resolution x
    :param resy: Video Resolution y
    :param dontcountarray: Cars that are not moving
    :return: Returns coordinates of the road along with area
    """

    xarray = []
    yarray = []

    box = {}
    for x in range(1, 10):
        for y in range(1, 10):
            key = f"{x, y}"
            box[key] = []
    ymax = resx
    xmax = resy

    cararea = 0
    for entry in dict:
        car = dict[entry]
        if entry not in dontcountarray:
            for dictframe in car:
                value = car[dictframe]
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
    coords = []
    try:
        results = ConvexHull(nparray)
        for simplex in results.simplices:
            coords.append((nparray[simplex, 0][0], nparray[simplex, 1][0]))
            coords.append((nparray[simplex, 0][1], nparray[simplex, 1][1]))
    except:
        coords = [(0,0),(0,0)]

    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    coords = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)


    ordered = []
    area = 0
    q = coords[-1]
    for i,entry in enumerate(coords):
        ordered.append([entry[0],entry[1]])
        if i % 2 == 0:

            area += entry[0]*q[1] - entry[1]*q[0]
            q = entry

    ordered.append([coords[0][0], coords[0][1]])

    return ordered, area



def correlation(carsdict, xyxy, framenumber, resx, resy, fps):
    """
    :param carsdict: Car Dictionary Consisting of all of the Detections by Vehicle Number
    :param xyxy: All of the detections in the current frame
    :param framenumber: Current frame number of video
    :param resx: Video Resolution x
    :param resy: Video Resolution y
    :param fps: Frames per second of video
    :return: Determines if a detection is a car by looking at the prior five frames and looking for cars that have been
    near that vehicle. Returns the cars dictionary with all of the cars in the given frame added to it
    """
    diffx = (xyxy[2]-xyxy[0])
    diffy = (xyxy[3]-xyxy[1])
    ratio = ((diffx+diffy)/2)/30
    threshold = 30*((resx*resy)/(1080*720))*(ratio)*(30/fps)
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



