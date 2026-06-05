#%%
import cv2
from bewegungserkennung import MotionDetector


class WebcamMotionApp:
    def __init__(self, camera_index=2):
        self.detector = MotionDetector()
        self.cap = cv2.VideoCapture(camera_index)

    def run(self):
        if not self.cap.isOpened():
            print("Fehler: Kamera konnte nicht geöffnet werden.")
            return

        print("Motion Detection gestartet – 'q' zum Beenden")

        ret, prev_frame = self.cap.read()
        if not ret:
            print("Fehler: Konnte ersten Frame nicht lesen.")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, curr_frame = self.cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Frames direkt übergeben statt über Instanzvariablen
            mask = self.detector.get_mask(prev_gray, curr_gray)
            self.detector.find_contour(mask)  # setzt self.objects intern

            output = self.detector.draw_detections(image=curr_frame.copy())

            n = len(self.detector.objects)
            if n > 0:
                cv2.putText(output, f"Bewegung: {n} Objekt(e)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Motion Detection", output)
            cv2.imshow("Maske", mask)

            prev_gray = curr_gray

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = WebcamMotionApp(camera_index=2)
    app.run()