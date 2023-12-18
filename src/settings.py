from typing import Dict, Tuple


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

    def handle_key(self, key: str) -> None:
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
