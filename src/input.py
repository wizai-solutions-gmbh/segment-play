from typing import Any

import cv2


class Interaction:
    def __init__(self) -> None:
        self.mouse_x = 0
        self.mouse_y = 0
        self.clicked = False

    def draw_mouse(
            self,
            event: int,
            x: int,
            y: int,
            flags: int,
            param: Any) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicked = True
            self.mouse_x = x
            self.mouse_y = y

    def check_clicked(self) -> bool:
        if self.clicked:
            self.clicked = False
            return True
        return False
