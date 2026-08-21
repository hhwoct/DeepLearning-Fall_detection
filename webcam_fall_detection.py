import cv2
import numpy as np
from collections import deque
import time
import os

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("请安装: pip install mediapipe==0.10.8")

class RuleBasedFallDetector:
    
    def __init__(self, fall_threshold=0.6):
        """
        Args:
            fall_threshold: 摔倒判定阈值 (0-1)
        """
        if not HAS_MEDIAPIPE:
            raise ImportError("请先安装mediapipe")
        
        # MediaPipe初始化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 高精度
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 参数
        self.fall_threshold = fall_threshold
        self.fall_score_history = deque(maxlen=15)  # 平滑
        
        # 状态
        self.fall_start_time = None
        self.is_falling = False
        self.fall_count = 0
        self.frame_count = 0
        
        # FPS
        self.fps_history = deque(maxlen=30)
        self.prev_time = time.time()
        
        print(f"✅ 初始化完成 (阈值: {fall_threshold:.0%})")
        print("控制: Q退出 | R重置 | +/-调阈值 | S截图\n")
    
    def calculate_fall_score(self, landmarks, h, w):

        # 提取关键点坐标
        def get_point(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
        # 关键点 (MediaPipe索引)
        nose = get_point(0)
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_hip = get_point(23)
        right_hip = get_point(24)
        left_knee = get_point(25)
        right_knee = get_point(26)
        left_ankle = get_point(27)
        right_ankle = get_point(28)
        
        # 计算身体中心线
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        knee_center = (left_knee + right_knee) / 2
        ankle_center = (left_ankle + right_ankle) / 2
        
        scores = []
        
        # === 特征1: 身体倾斜角度 ===
        body_vector = shoulder_center - hip_center
        vertical = np.array([0, -1])  # 垂直向上
        
        # 计算身体向量与垂直方向的夹角
        cos_angle = np.dot(body_vector, vertical) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        # 角度越大，越可能摔倒
        if angle > 60:
            scores.append(1.0)
        elif angle > 45:
            scores.append(0.8)
        elif angle > 30:
            scores.append(0.5)
        elif angle > 15:
            scores.append(0.2)
        else:
            scores.append(0.0)
        
        # === 特征2: 肩膀-髋部高度比 ===
        shoulder_height = shoulder_center[1]
        hip_height = hip_center[1]
        body_height = abs(shoulder_height - hip_height)
        
        # 正常站立时身体高度应该较大
        if body_height < 30:  # 像素值
            scores.append(1.0)
        elif body_height < 60:
            scores.append(0.7)
        elif body_height < 100:
            scores.append(0.3)
        else:
            scores.append(0.0)
        
        # === 特征3: 髋部位置（是否接近地面） ===
        ground_ratio = hip_height / h
        if ground_ratio > 0.75:  # 髋部在下半部分
            scores.append(1.0)
        elif ground_ratio > 0.65:
            scores.append(0.6)
        elif ground_ratio > 0.55:
            scores.append(0.3)
        else:
            scores.append(0.0)
        
        # === 特征4: 宽高比 ===
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        if body_height > 0:
            aspect_ratio = shoulder_width / body_height
            if aspect_ratio > 2.0:  # 宽大于高
                scores.append(1.0)
            elif aspect_ratio > 1.5:
                scores.append(0.7)
            elif aspect_ratio > 1.0:
                scores.append(0.4)
            else:
                scores.append(0.0)
        
        # === 特征5: 头部位置 ===
        head_ground_ratio = nose[1] / h
        if head_ground_ratio > 0.7:  # 头接近地面
            scores.append(1.0)
        elif head_ground_ratio > 0.5:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # === 特征6: 膝盖位置 ===
        knee_ground_ratio = knee_center[1] / h
        if knee_ground_ratio > 0.9:  # 膝盖接近地面
            scores.append(0.8)
        elif knee_ground_ratio > 0.7:
            scores.append(0.4)
        else:
            scores.append(0.0)
        
        # 综合分数（加权平均）
        weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # 特征权重
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        return final_score, {
            'angle': angle,
            'body_height': body_height,
            'hip_ratio': ground_ratio,
            'aspect_ratio': shoulder_width / (body_height + 1e-6),
            'head_ratio': head_ground_ratio
        }
    
    def process_frame(self, frame):
        """处理一帧"""
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # MediaPipe处理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks is None:
            # 没有检测到人
            self.draw_no_person(display)
            return display, False, 0.0
        
        # 计算摔倒分数
        fall_score, details = self.calculate_fall_score(
            results.pose_landmarks.landmark, h, w
        )
        
        # 平滑处理
        self.fall_score_history.append(fall_score)
        smoothed_score = np.mean(self.fall_score_history)
        
        # 判定摔倒
        is_fall = smoothed_score > self.fall_threshold
        
        # 摔倒持续时间判定（避免误报）
        if is_fall:
            if self.fall_start_time is None:
                self.fall_start_time = time.time()
            elif time.time() - self.fall_start_time > 1.0:  # 持续1秒
                if not self.is_falling:
                    self.is_falling = True
                    self.fall_count += 1
                    print(f"🚨 检测到摔倒！(第{self.fall_count}次) 分数:{smoothed_score:.1%}")
        else:
            self.fall_start_time = None
            self.is_falling = False
        
        # 绘制结果
        self.draw_results(display, results.pose_landmarks, smoothed_score, is_fall, details)
        
        return display, is_fall, smoothed_score
    
    def draw_no_person(self, frame):
        """无人时的显示"""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (80, 80, 80), -1)
        cv2.putText(frame, "No Person Detected", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_results(self, frame, landmarks, fall_score, is_fall, details):
        """绘制检测结果"""
        h, w = frame.shape[:2]
        
        # 绘制骨架
        self.mp_draw.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
        
        # 状态栏
        if is_fall:
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 255), -1)
            cv2.putText(frame, "⚠️ FALL DETECTED!", (w//2-180, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 100, 0), -1)
            # 分数颜色
            if fall_score < 0.3:
                color = (0, 255, 0)
                status = "Standing"
            elif fall_score < 0.6:
                color = (0, 255, 255)
                status = "Warning"
            else:
                color = (0, 165, 255)
                status = "Risk"
            
            cv2.putText(frame, f"Status: {status}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Score: {fall_score:.1%}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 信息面板
        panel_x, panel_y = 10, h - 160
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+300, h-10), 
                     (30, 30, 30), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+300, h-10),
                     (100, 100, 100), 1)
        
        y_offset = panel_y + 25
        info_items = [
            f"Score: {fall_score:.1%}",
            f"Angle: {details.get('angle', 0):.0f} deg",
            f"Hip Pos: {details.get('hip_ratio', 0):.1%}",
            f"Aspect: {details.get('aspect_ratio', 0):.2f}",
            f"Fall Count: {self.fall_count}"
        ]
        
        for item in info_items:
            cv2.putText(frame, item, (panel_x+15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        # 底部概率条
        bar_y = h - 25
        bar_w = w - 200
        bar_x = 10
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+15), (80, 80, 80), -1)
        
        # 分数条
        fill_w = int(bar_w * fall_score)
        if fall_score > self.fall_threshold:
            bar_color = (0, 0, 255)
        elif fall_score > 0.4:
            bar_color = (0, 255, 255)
        else:
            bar_color = (0, 255, 0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+15), bar_color, -1)
        
        # 阈值线
        thresh_x = bar_x + int(bar_w * self.fall_threshold)
        cv2.line(frame, (thresh_x, bar_y-3), (thresh_x, bar_y+18), (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold:{self.fall_threshold:.0%}", 
                   (bar_x+bar_w+10, bar_y+12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            t = time.time()
            self.fps_history.append(10 / (t - self.prev_time + 0.001))
            self.prev_time = t
        
        if self.fps_history:
            fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {fps:.0f}", (w-100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def run(self, camera_id=0):
        """运行实时检测"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ 无法打开摄像头！")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("📹 摄像头已打开\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                display, is_fall, score = self.process_frame(frame)
                cv2.imshow('Fall Detection - Rule Based', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.fall_count = 0
                    self.fall_score_history.clear()
                    self.fall_start_time = None
                    self.is_falling = False
                    print("🔄 已重置")
                elif key in [ord('+'), ord('=')]:
                    self.fall_threshold = min(0.99, self.fall_threshold + 0.05)
                    print(f"阈值: {self.fall_threshold:.0%}")
                elif key == ord('-'):
                    self.fall_threshold = max(0.1, self.fall_threshold - 0.05)
                    print(f"阈值: {self.fall_threshold:.0%}")
                elif key == ord('s'):
                    name = f"fall_{time.strftime('%H%M%S')}.jpg"
                    cv2.imwrite(name, display)
                    print(f"📸 {name}")
        
        except KeyboardInterrupt:
            print("\n⏹ 用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n✅ 共检测到 {self.fall_count} 次摔倒事件")


if __name__ == '__main__':
    print("=" * 50)
    print("  实时摔倒检测系统 (规则版本)")
    print("=" * 50 + "\n")
    
    # 阈值0.5-0.6效果最好
    detector = RuleBasedFallDetector(fall_threshold=0.55)
    detector.run(0)