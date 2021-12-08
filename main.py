import pyzed.sl as sl
import cv2
import numpy as np
import time
import math


class YolowZED:

    def __init__(self,GPU_Switch):
        self.GPU_Switch = GPU_Switch
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()

        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
        self.tr_np = mirror_ref.m

    def load_paths(self):
        self.weightsPath = 'C:\\Users\\Erenk\\Desktop\\py\\zed\\yolov4-parfume_last.weights'
        self.configPath = 'C:\\Users\\Erenk\\Desktop\\py\\zed\\yolov4-parfume.cfg'
        labelsPath = 'C:\\Users\\Erenk\\Desktop\\py\\zed\\parfume.names'
        self.labels = open(labelsPath).read().strip().split("\n") 


    def initZED(self):
            self.zed = sl.Camera()

            init_params = sl.InitParameters()
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
            init_params.coordinate_units = sl.UNIT.METER 
            init_params.camera_resolution = sl.RESOLUTION.HD720

            err = self.zed.open(init_params)
    
            if err != sl.ERROR_CODE.SUCCESS:
                exit(1)

            self.runtime_parameters = sl.RuntimeParameters()
            self.runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  
            self.runtime_parameters.confidence_threshold = 100
            self.runtime_parameters.textureness_confidence_threshold = 100


    def getMeasure_of_pix(self,x,y):
        err, point_cloud_value = self.point_cloud.get_value(x, y)    
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                            point_cloud_value[1] * point_cloud_value[1] +
                            point_cloud_value[2] * point_cloud_value[2])   
        point_cloud_np = self.point_cloud.get_data()
        point_cloud_np.dot(self.tr_np)

        return distance     
    
    def main(self):

        self.load_paths()
        self.initZED()

        num_frames = 1
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        if self.GPU_Switch == 1:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        count = 0
        sum_of_fps = 0
        while True:
            points = []

            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS: 
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
                frame = self.image.get_data()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{self.labels[classIDs[i]]}: {round(confidences[i], 2)}",(x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


                    center_X = int(x + (w / 2))
                    center_Y = int(y + (h / 2))
                    cv2.circle(frame, (center_X, center_Y), 1, (0, 0, 255), thickness=2) # center of object

                    distance = (self.getMeasure_of_pix(center_X,center_Y))
                    cv2.putText(frame, f"{round(distance,3)} m",(x, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            

            end = time.time()
            seconds = end - start
            fps = num_frames / seconds
            cv2.putText(frame,f"FPS: {fps}",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)

            sum_of_fps += fps
            count +=1
            cv2.imshow("frame",frame)

            if cv2.waitKey(1) == ord('q'):
                break

        print(round(sum_of_fps/count,3)) # average fps
        self.zed.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    process = YolowZED(1)
    process.main()