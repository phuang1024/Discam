"""
PTZ camera control via Pelco-D protocol via USB serial port.
"""

import time

import serial


class PTZControl:
    def __init__(self):
        self.port = serial.Serial("/dev/ttyUSB1")
        self.addr = 1

    def set_velocity(self, pan: float, tilt: float, zoom: float):
        """
        Set PTZ velocity.

        pan, tilt, zoom are floats in [-1, 1]. +-1 corresponds to max velocity.

        Positive zoom is zoom in.
        """
        # Calculate speeds.
        pan_spd = min(int(abs(pan) * 0x3F), 0x3F)
        tilt_spd = min(int(abs(tilt) * 0x3F), 0x3F)
        zoom_spd = min(int(abs(zoom) * 0x33), 0x33)

        # Set zoom speed.
        #self.send_command(0, 0x25, 0, zoom_spd)

        # Send PTZ command.
        byte4 = 0
        byte4 |= 32 if zoom > 0 else 64
        byte4 |= 16 if tilt > 0 else 8
        byte4 |= 2 if pan > 0 else 4
        self.send_command(0, byte4, pan_spd, tilt_spd)

    def send_command(self, byte3, byte4, byte5, byte6):
        """
        Use default address. Computes checksum.
        """
        command = [0xFF, self.addr, byte3, byte4, byte5, byte6]
        command.append(sum(command[1:]) % 256)
        command = bytes(command)
        self.port.write(command)
        self.port.flush()


if __name__ == "__main__":
    ptz = PTZControl()
    ptz.set_velocity(0.3, 0, 0)
    time.sleep(1)
    ptz.set_velocity(-0.3, 0, 0)
    time.sleep(1)
    ptz.set_velocity(0, 0, 0)
