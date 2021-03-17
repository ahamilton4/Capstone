import argparse
from sys import platform

from utils.models import *  # set ONNX_EXPORT in models.py
from utils.utils import *
from utils.datasets import LoadStreams, LoadImages
import Calculations

def detect(save_txt=False, save_img=False):
    img_size = (608, 352) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    cardict = {}
    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=10)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)

    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    framecount = 0
    for path, img, im0s, vid_cap in dataset:
        numframes = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            fps = round(vid_cap.get(cv2.CAP_PROP_FPS),0)
        except:
            fps = 30

        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):
            framecount += 1# detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                alldetects = []
                for *xyxy, conf, cls in det:
                    xyxy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    alldetects.append(xyxy)
                    cardict, carnum = Calculations.correlation(cardict, xyxy, framecount, im0.shape[0], im0.shape[1],fps)
                    if 1==1:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '#%s' % (carnum)

            dcarray = Calculations.validcars(cardict,im0.shape[0],im0.shape[1], framecount)
            ch, roadarea = Calculations.road(cardict,im0.shape[0], im0.shape[1],dcarray)
            numcars, cararea, xyxy = Calculations.carsonroad(alldetects,ch)
            flowin,flowout = Calculations.flowrate(cardict,fps, dcarray, 1, framecount, numcars)

            flowin = round(flowin, 2)
            flowout = round(flowout,2)


            for det in xyxy:
                plot_one_box(det,im0,color = (255,0,0))
            try:
                density = cararea / (roadarea / 2) * 100
            except:
                density = 0.0

            if density > 100:
                density = 0

            hull = np.array(ch)
            blk = np.zeros(im0.shape,np.uint8)
            try:
                cv2.fillPoly(blk,np.int32([hull]),(255,255,255))
            except:
                print("")
            output = cv2.addWeighted(im0,1.0,blk,0.4,1)

            x = im0.shape[1]
            y = im0.shape[0]
            # c1 = tuple([int(im0.shape[0] - (im0.shape[0]/4)),int(im0.shape[1])])
            # c2 = tuple([int(im0.shape[0]),int(im0.shape[1] - (im0.shape[1]/4))])
            c1box = tuple([x , int(y - (y / 4))])
            c2box = tuple([int(x - (x / 4)) , y])

            cv2.rectangle(output, c1box, c2box, (0, 0, 0), thickness=cv2.FILLED)

            calculations = ['DENSITY', 'COUNT', 'FLOW IN', 'FLOW OUT']
            results = [density, numcars, flowin, flowout]

            for index, calculation in enumerate(calculations):

                xlabel = int(x - (x/4))
                ylabel = int(y - (y/5) + (20 * (index))+20)
                string = f"{calculation}:"
                org = tuple([xlabel,ylabel])
                cv2.putText(img=output, text=string, org=org, fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1.0,color=(255,255,255),thickness=1)

                xlabel = int(x - (x / 8))
                org = tuple([xlabel,ylabel])
                if index == 0:
                    if results[0] != 0.0 and framecount > numframes * .2:
                        string = f"{round(results[0],2)} %"
                    else:
                        string = "CALCULATING.."
                elif index == 1:
                    if results[1] != 0 and framecount > numframes * .2:
                        string = f"{results[index]} CARS"
                    else:
                        string = "CALCULATING.."
                else:
                    if flowout != 0 and framecount > numframes * .2:
                        string = f"{results[index]} CARS/MINUTE"
                    else:
                        string = "CALCULATING.."
                cv2.putText(img=output, text=string, org=org, fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1.0,color=(255,255,255),thickness=1)

            boxes = 20
            width = (x/4)/boxes
            for i in range(boxes):
                y1= int(y-(y/4))
                y2= int(y1 + width-2)
                if framecount > (numframes * (i+1)/boxes):
                    x1 = int(x-(x/4)+(i*width))
                    x2 = int(x1 + width-2)
                    c1 = tuple([x1,y1])
                    c2 = tuple([x2,y2])
                    cv2.rectangle(output,c1,c2,(255,255,255),thickness=cv2.FILLED)

            print('%sDone. (%.3fs)' % (s, time.time() - t))
            # print("YOOYOYO",framecount)
            #print(cardict)
            # Stream results
            if view_img:
                cv2.imshow(p, output)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, output)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(output)
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)
    print(cardict)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
