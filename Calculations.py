from math import sqrt

def density(xyxy):

    return

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
    if validated == False:
        length = len(carsdict)
        carsdict[str(length+1)] = {str(framenumber):xyxy}
        carnum = length + 1
    return carsdict, carnum


