import numpy as np
from ultralytics import YOLO

class Tracker:
    def __init__(self, roi_padding, roi_ratio):
        self.roi_padding = roi_padding
        self.roi_ratio = roi_ratio
    
    def _default_roi(self, frame):
        # 짧은 차원을 기준으로 ROI 크기 결정
        height, width = frame.shape[:2]
        roi_size = int(min(width, height) * self.roi_ratio)
        
        # 화면 중심 계산
        center_x = width // 2
        center_y = height // 2
        
        # ROI 좌표 계산 (정수 변환)
        x1 = int(max(0, center_x - roi_size // 2))
        y1 = int(max(0, center_y - roi_size // 2))
        x2 = int(min(width, x1 + roi_size)) # x1 기준 크기 더하기
        y2 = int(min(height, y1 + roi_size)) # y1 기준 크기 더하기
        
        return (x1, y1, x2, y2)

    def _calculate_roi(self, frame, landmarks):
        height, width = frame.shape[:2]
        
        # 랜드마크가 없거나 빈 리스트인 경우 기본 ROI 계산
        if not landmarks:
            # print("No landmarks provided, calculating default ROI.") # 디버깅용
            return self._default_roi(frame)
            
        # Normalized 좌표를 픽셀 좌표로 변환
        x_values = [lm.x * width for lm in landmarks]
        y_values = [lm.y * height for lm in landmarks]
        
        # 랜드마크 기반 ROI 좌표 계산
        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)
        
        # 패딩 추가
        x_min = int(x_min - self.roi_padding)
        y_min = int(y_min - self.roi_padding)
        x_max = int(x_max + self.roi_padding)
        y_max = int(y_max + self.roi_padding)
        
        # 경계 검사 (ROI가 프레임을 벗어나는 경우 조정)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        # 유효한 박스인지 확인 (넓이/높이가 0보다 커야 함)
        if x_min < x_max and y_min < y_max:
            return (x_min, y_min, x_max, y_max)
        else:
            return self._calculate_default_roi(width, height)

        
class yoloTracker:
    def __init__(self, roi_padding=20, choice_area=0.7, iou_treshold=0.1, adjust_rate=0.1):
        self.model = YOLO("../checkpoint/yolov9t.pt")
        self.roi_padding = roi_padding
        self.adjust_rate = adjust_rate
        self.choice_area = choice_area
        self.iou_treshold = iou_treshold
        
    def _calulate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area
        
    def _choice_target(self, height, width, rois):
        [xm, ym] = [width/2, height/2]
        add_size = (min([height, width])*self.choice_area)/2
        
        area = [int(max(0, xm-add_size)),
               int(max(0, ym-add_size)),
               int(min(width, xm+add_size)),
               int(min(height, ym+add_size)),]
        
        max_iou, target_id = 0, None
        
        for roi in rois:
            roi_id = int(roi[0])
            x1, y1, x2, y2 = roi[1:5]
            iou = self._calulate_iou(area, [x1, y1, x2, y2])
            if iou > max_iou:
                max_iou, target_id = iou, roi_id
                
        if max_iou < self.iou_treshold:
            return None
            
        print("track id is : ", target_id)
        return target_id
        
    def _roi_scaler(self, height, width, roi):
        x1, y1, x2, y2 =  roi
        roi_width, roi_height = x2-x1, y2-y1
        
        if roi_height > roi_width :
            add_len = (roi_height-roi_width)/2
            x1 -= add_len/3
            x2 += add_len/3
        else:
            add_len = (roi_width-roi_height)/2
            y1 -= add_len/3
            y2 += add_len/3
            
        padding = int(max(0, (self.roi_padding - 20)))
        
        x1 = x1 - padding
        y1 = y1 - padding 
        x2 = x2 + padding 
        y2 = y2 + padding 
        
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(width, x2))
        y2 = int(min(height, y2))
        
        return (x1, y1, x2, y2)
        
    def computed_roi(self, frame, target_reset, target_id):
        height, width = frame.shape[:2]
        
        result = self.model.track(frame, persist=True, stream=False, verbose=False, conf=0.8)[0]

        if result.boxes.id is None or len(result.boxes) == 0:
            return None, target_id
            
        try:
            ids = np.array(result.boxes.id.int().cpu().numpy()).reshape(-1, 1)
            bboxes = np.array(result.boxes.xyxy.cpu().numpy())
            labels = np.array(result.boxes.cls.cpu().numpy().astype(int)).reshape(-1, 1)
            
            output = np.hstack((ids, bboxes, labels))
            filtered_output = output[output[:, -1] == 0]
            
            if target_reset:
                target_id = self._choice_target(height, width, filtered_output)
                
            if target_id != None:
                target_bbox = filtered_output[filtered_output[:, 0] == target_id]
                
                if len(target_bbox)==0 :
                    return None, target_id
                else:
                    roi = self._roi_scaler(height, width, target_bbox[0][1:5])
                    return roi, target_id
                    
            else: return None, target_id
        except Exception as e:
            print(f"[WARN] mmTracker ROI 실패: {e}")
            return None, target_id
            
    def adjust_roi(self, frame, landmarks, former_roi):
        height, width = frame.shape[:2]
        roi = ()
        
        if not landmarks:
            return None
            
        try:
            x_values = [lm.x * width for lm in landmarks]
            y_values = [lm.y * height for lm in landmarks]
            
            if not x_values or not y_values:
                 return None
                
        except AttributeError:
             print("Error: Landmarks object structure unexpected.")
             return None
        
        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)
        
        unscaled_roi = (x_min, y_min, x_max, y_max)
        present_roi = self._roi_scaler(height, width, unscaled_roi)
        
        for present, former in zip(present_roi, former_roi):
            adjusted = self.adjust_rate * present + (1 - self.adjust_rate) * former
            roi = roi + (int(adjusted), )
            
        return roi