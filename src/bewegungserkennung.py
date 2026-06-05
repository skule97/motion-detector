#%%
import cv2
import numpy as np

class MotionDetector:
    
    def __init__(self, kernel_size=(9, 9), threshold=20, area_threshold=500):
        self.kernel = np.ones(kernel_size, dtype=np.uint8)
        self.threshold = threshold
        self.area_threshold = area_threshold
        self.objects = []
    

    def get_frames(self,frame1,frame2):
        im1=cv2.imread(frame1)
        im2=cv2.imread(frame2)

        self.img1_gray=cv2.cvtColor(im1,cv2 .COLOR_BGR2GRAY)
        self.img2_gray=cv2.cvtColor(im2,cv2 .COLOR_BGR2GRAY)

        return self.img1_gray,self.img2_gray



    def get_mask(self,frame1=None,frame2=None):

        if frame1 is None or frame2 is None:
            frame1, frame2 = self.img1_gray, self.img2_gray
        
        frame_diff=cv2.absdiff(frame2,frame1)
        
        #macht aus Bild binäres Bild(graustufen zu schwart und weiß)
        #_ ist der Verwendete SChwellwert( hier 20)
        _, mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
        
        #schaut sich in 3x3 Feld an ob ein PIxel weiß oder schwarz ist und entscheidet dann ob es weiß oder schwarz ist, dadurch werden kleine Störungen entfernt
        mask = cv2.medianBlur(mask, 3)     

    # Morphologische Operationen: Lücken schließen + ausdehnen
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1) #macht Löcher zu
        mask = cv2.dilate(mask, self.kernel, iterations=4)  #macht objkete größer, damit sie besser erkannt werden können

        return mask


    def find_contour(self, mask=None):
        if mask is None:
            mask = self.get_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        detected_objects = []
        scores = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > self.area_threshold:
                detected_objects.append((x, y, w, h))
                scores.append(area)  # Fläche als Score für NMS

        if len(detected_objects) == 0:
            self.objects = []
            return np.array([])

        boxes = np.array(detected_objects)
        scores = np.array(scores, dtype=float)
        self.objects = self.non_max_suppression(boxes, scores)
        return self.objects

    
    def draw_detections(self, image=None, output_path=None):
        if image is None:
            image = self.img2_gray.copy()
        for (x, y, w, h) in self.objects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if output_path:
            cv2.imwrite(output_path, image)

        return image

    def _remove_contained_boxes(self, objects):
        """Entfernt Boxen, die vollständig in einer anderen Box enthalten sind."""
        check_array = np.array([True, True, False, False])
        keep = list(range(len(objects)))
        for i in keep:
            for j in range(len(objects)):
                if np.all((np.array(objects[j]) >= np.array(objects[i])) == check_array):
                    try:
                        keep.remove(j)
                    except ValueError:
                        continue
        return keep
    

    def non_max_suppression(self, boxes, scores, threshold=1e-1):
        """Filtert überlappende Bounding Boxes mittels Non-Maximum Suppression."""
        boxes = boxes[np.argsort(scores)[::-1]]
        order = self._remove_contained_boxes(boxes)
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            for j in order[:]:  # Kopie iterieren, damit remove sicher ist
                x1 = max(boxes[i][0], boxes[j][0])
                y1 = max(boxes[i][1], boxes[j][1])
                x2 = min(boxes[i][0] + boxes[i][2], boxes[j][0] + boxes[j][2])
                y2 = min(boxes[i][1] + boxes[i][3], boxes[j][1] + boxes[j][3])
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area_i = boxes[i][2] * boxes[i][3]
                area_j = boxes[j][2] * boxes[j][3]
                union = area_i + area_j - intersection
                iou = intersection / union if union > 0 else 0
                if iou > threshold:
                    order.remove(j)
        return boxes[keep]
# %%
