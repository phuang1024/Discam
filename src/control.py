"""
PTZ camera control via Pelco-D protocol via USB serial port.
"""

import time

import serial


class PTZControl:
    def __init__(self):
        self.port = serial.Serial("/dev/ttyUSB0")
        self.addr = 1

    def set_pt(self, pan: int, tilt: int):
        """
        Set pan and tilt velocity.

        pan, tilt: -1, 0, or 1.
        """
        byte4 = 0
        if tilt > 0:
            byte4 |= 16
        elif tilt < 0:
            byte4 |= 8
        if pan > 0:
            byte4 |= 2
        elif pan < 0:
            byte4 |= 4
        self.send_command(0, byte4, 0x10, 0x10)

    def set_zoom(self, zoom: int):
        """
        Set zoom velocity.

        zoom: -1, 0, or 1.
        """
        byte4 = 0
        if zoom > 0:
            byte4 |= 0x20
        elif zoom < 0:
            byte4 |= 0x40
        self.send_command(0, byte4, 0, 0)

    def stop(self):
        """
        Stop all motion.
        """
        self.send_command(0, 0, 0, 0)

    def send_command(self, byte3, byte4, byte5, byte6):
        """
        Use default address. Computes checksum.
        """
        command = [0xFF, self.addr, byte3, byte4, byte5, byte6]
        command.append(sum(command[1:]) % 256)
        print(command)
        command = bytes(command)
        self.port.write(command)
        self.port.flush()


if __name__ == "__main__":
    ptz = PTZControl()
    for i in range(10):
        ptz.set_pt(1, 0)
        time.sleep(1)
        ptz.set_pt(-1, 0)
        time.sleep(1)
        ptz.stop()

        ptz.set_zoom(1)
        time.sleep(1)
        ptz.set_zoom(-1)
        time.sleep(1)
        ptz.stop()
        break
