from typing import Dict, Optional, Tuple

from util.controller import XboxController


class GameSettings:
    save_imgs: bool = False
    overall_mirror: bool = True
    random_people_mirror: bool = False
    form_invisibility: bool = False
    all_invisibility: bool = False
    hide_background: bool = False
    gray_game: bool = False
    black: bool = False
    show_boxes: bool = False
    show_poses: bool = False
    id_position_map: Dict[int, float] = {}
    segmentation_parts: int = 1
    segmentation_change: bool = False

    def __init__(self, controller: bool = True) -> None:
        self.controller: Optional[XboxController] = None
        self.controller_keypressed: Dict[str, bool] = {}
        if controller:
            self.controller = XboxController()

    def update(self) -> None:
        if not self.controller:
            return
        if self.controller.A == 1:
            self.controller_keypressed['A'] = True
        elif self.controller_keypressed.get('A', False):
            self.controller_keypressed['A'] = False
            self.all_invisibility = not self.all_invisibility
        if self.controller.X == 1:
            self.controller_keypressed['X'] = True
        elif self.controller_keypressed.get('X', False):
            self.controller_keypressed['X'] = False
            self.overall_mirror = not self.overall_mirror
        if self.controller.Y == 1:
            self.controller_keypressed['Y'] = True
        elif self.controller_keypressed.get('Y', False):
            self.controller_keypressed['Y'] = False
            self.form_invisibility = not self.form_invisibility
        if self.controller.B == 1:
            self.controller_keypressed['B'] = True
        elif self.controller_keypressed.get('B', False):
            self.controller_keypressed['B'] = False
            self.hide_background = not self.hide_background

        if self.controller.Back == 1:
            self.controller_keypressed['Back'] = True
        elif self.controller_keypressed.get('Back', False):
            self.controller_keypressed['Back'] = False
            self.reset()

        if self.controller.LeftTrigger > 0:
            self.controller_keypressed['LeftTrigger'] = True
        elif self.controller_keypressed.get('LeftTrigger', False):
            self.controller_keypressed['LeftTrigger'] = False
            self.random_people_mirror = not self.random_people_mirror
        if self.controller.RightTrigger > 0:
            self.controller_keypressed['RightTrigger'] = True
        elif self.controller_keypressed.get('RightTrigger', False):
            self.controller_keypressed['RightTrigger'] = False
            self.gray_game = not self.gray_game
            self.id_position_map = {}

        if self.controller.RightBumper == 1:
            self.controller_keypressed['RightBumper'] = True
        elif self.controller_keypressed.get('RightBumper', False):
            self.controller_keypressed['RightBumper'] = False
            self.show_boxes = not self.show_boxes
        if self.controller.LeftBumper == 1:
            self.controller_keypressed['LeftBumper'] = True
        elif self.controller_keypressed.get('LeftBumper', False):
            self.controller_keypressed['LeftBumper'] = False
            self.show_poses = not self.show_poses

        if self.controller.LeftBumper == 1:
            self.controller_keypressed['LeftBumper'] = True
        elif self.controller_keypressed.get('LeftBumper', False):
            self.controller_keypressed['LeftBumper'] = False
            self.show_poses = not self.show_poses

        if self.controller.LeftThumb == 1:
            self.controller_keypressed['LeftThumb'] = True
        elif self.controller_keypressed.get('LeftThumb', False):
            self.controller_keypressed['LeftThumb'] = False
            self.segmentation_parts = 0
            self.segmentation_change = True

        if self.controller.LeftJoystickX > 0.5:
            self.controller_keypressed['LeftJoystickRight'] = True
        elif self.controller_keypressed.get('LeftJoystickRight', False):
            self.controller_keypressed['LeftJoystickRight'] = False
            self.segmentation_parts = 2
            self.segmentation_change = True

        if self.controller.LeftJoystickX < -0.5:
            self.controller_keypressed['LeftJoystickLeft'] = True
        elif self.controller_keypressed.get('LeftJoystickLeft', False):
            self.controller_keypressed['LeftJoystickLeft'] = False
            self.segmentation_parts = 0
            self.segmentation_change = True

        if self.controller.LeftJoystickY < -0.5:
            self.controller_keypressed['LeftJoystickDown'] = True
        elif self.controller_keypressed.get('LeftJoystickDown', False):
            self.controller_keypressed['LeftJoystickDown'] = False
            self.segmentation_parts = 3
            self.segmentation_change = True

        if self.controller.LeftJoystickY > 0.5:
            self.controller_keypressed['LeftJoystickUp'] = True
        elif self.controller_keypressed.get('LeftJoystickUp', False):
            self.controller_keypressed['LeftJoystickUp'] = False
            self.segmentation_parts = 1
            self.segmentation_change = True

    def handle_key(self, key: str) -> None:
        self.update()

        if key == 'm':
            self.overall_mirror = not self.overall_mirror
        if key == 'r':
            self.random_people_mirror = not self.random_people_mirror
        if key == 'f':
            self.form_invisibility = not self.form_invisibility
        if key == 'i':
            self.all_invisibility = not self.all_invisibility
        if key == 'j':
            self.hide_background = not self.hide_background
        if key == 'b':
            self.black = not self.black
        if key == 'g':
            self.gray_game = not self.gray_game
            self.id_position_map = {}
        if key == 's':
            self.show_boxes = not self.show_boxes
        if key == 'p':
            self.show_poses = not self.show_poses

        if key == '1':
            self.segmentation_parts = 0
            self.segmentation_change = True
        if key == '2':
            self.segmentation_parts = 1
            self.segmentation_change = True
        if key == '3':
            self.segmentation_parts = 2
            self.segmentation_change = True
        if key == '4':
            self.segmentation_parts = 3
            self.segmentation_change = True
        if key == '5':
            self.segmentation_parts = 4
            self.segmentation_change = True

        if key == 'o':
            self.reset()

    def reset(self) -> None:
        self.overall_mirror = True
        self.random_people_mirror = False
        self.form_invisibility = False
        self.all_invisibility = False
        self.hide_background = False
        self.gray_game = False
        self.black = False
        self.show_boxes = False
        self.show_poses = False
        self.id_position_map = {}
        self.segmentation_parts = 0
        self.segmentation_change = True

    def check_segmentation(self) -> Tuple[bool, int]:
        changed = self.segmentation_change
        self.segmentation_change = False
        return changed, self.segmentation_parts

    def print(self) -> None:
        print(
            'Controls:'
            + '\nPress "f" for hiding people based on their area covering the image or pose data.'  # noqa: E501
            + '\nPress "i" for triggering invisibility of all detected people.'
            + '\nPress "j" for hiding everything except all detected people.'
            + '\nPress "g" for modifying color of all detected people based on their position.'  # noqa: E501
            + '\nPress "r" for randomly mirror position and mask of some people.'  # noqa: E501
            + '\nPress "b" for triggering a black screen.'
            + '\nPress "m" for mirroring the image.'
            + '\nPress "s" for rendering bounding boxes of detected people.'
            + '\nPress "p" for rendering poses of detected people.'
            + '\nPress "o" for resetting all settings.'
        )
