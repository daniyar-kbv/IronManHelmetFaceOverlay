import cv2
import os


class FaceMasker:
    recording_fps = 30

    __faceCascade = cv2.CascadeClassifier(os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml")
    __video_capture = None
    __scale = 1
    __video_writer = None

    # faces recognition
    def __get_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.__faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

    def __mask(self, frame, faces, mask):
        # resize the mask to face size
        def resize(image, size):
            r = size / image.shape[1]
            dim = (size, int(image.shape[0] * r))
            return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        # creating the mask using thresholding
        def get_inv_thresh(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV)
            return thresh

        # applying the mask to the frame
        def merge(image, thresh, mask):
            background = image
            background = cv2.bitwise_and(background, background, mask=thresh)
            return cv2.bitwise_or(background, mask)

        for (x,y,w,h) in faces:
            scaled = int(w * self.__scale)
            resized_mask = resize(mask, scaled)

            # rescale the image origin to cover the whole face
            offset = int(((scaled - w) / 2))
            x1, y1, = x - offset, y - offset
            x2, y2 = x1 + scaled, y1 + scaled

            # if the mask is inside the frame
            if 0 < x1 < frame.shape[1] and \
                0 < x2 < frame.shape[1] and \
                0 < y1 < frame.shape[0] and \
                0 < y2 < frame.shape[0]:
                thresh = get_inv_thresh(resized_mask)
                frame[y1:y2, x1:x2] = merge(frame[y1:y2, x1:x2], thresh, resized_mask)

        return frame

    # create video writer for recording
    def prepare_recording(self, path):
        width = self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.__video_writer = cv2.VideoWriter(path, fourcc, self.recording_fps, (int(width), int(height)))

    # main method for the usage
    def start(self, mask, scale, recording_path=None):
        self.__scale = scale
        self.__video_capture = cv2.VideoCapture(0)

        if recording_path:
            self.prepare_recording(recording_path)

        while self.__video_capture.isOpened():
            _, frame = self.__video_capture.read()
            frame = cv2.flip(frame, 1)
            faces = self.__get_faces(frame)
            frame = self.__mask(frame, faces, mask)

            if recording_path:
                self.__video_writer.write(frame)

            cv2.imshow('Face Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if recording_path:
            self.__video_writer.release()

        self.__video_capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

mask = cv2.imread('iron_man_helmet.png')
face_masker = FaceMasker()
face_masker.start(mask, 1.8, 'Daniyar_Kurmanbayev_Task5_CartoonFilter.mp4')
