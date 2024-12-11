import cv2


class VideoReader:
    """
    Video reader with buffer of the next N frames.
    Buffer is list of frames, with the first index being the oldest.
    """

    def __init__(self, file, res: int, buffer_size):
        """
        res: Width resolution of frames.
        """
        self.video = cv2.VideoCapture(file)
        self.res = res
        self.buffer_size = buffer_size

        self.buffer = []
        for i in range(buffer_size):
            frame = self._read_frame()
            self.buffer.append(frame)

    def __getitem__(self, index):
        return self.buffer[index]

    def next(self):
        """
        Read next frame and add to end of buffer.
        Return this frame.
        """
        frame = self._read_frame()
        self.buffer.pop(0)
        self.buffer.append(frame)
        return frame

    def _read_frame(self):
        ret, frame = self.video.read()
        if not ret:
            raise ValueError("No more frames to read")
        height = int(frame.shape[0] * self.res / frame.shape[1])
        frame = cv2.resize(frame, (self.res, height))
        return frame
