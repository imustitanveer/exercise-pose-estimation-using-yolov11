import numpy as np
import cv2
from ultralytics.utils.plotting import Annotator, colors
from copy import deepcopy

def plot(pose_result, plot_size_redio, show_points=None, show_skeleton=None):
    class _Annotator(Annotator):
        def kpts(self, kpts, shape=(640, 640), radius=5, line_thickness=2, kpt_line=True):
            if self.pil:
                self.im = np.asarray(self.im).copy()
            nkpt, ndim = kpts.shape
            is_pose = nkpt == 17 and ndim == 3
            kpt_line &= is_pose
            color_palette = colors
            for i, k in enumerate(kpts):
                if show_points is not None and i not in show_points:
                    continue
                if len(k) == 3 and k[2] < 0.5:
                    continue
                cv2.circle(self.im, (int(k[0]), int(k[1])), int(radius * plot_size_redio),
                           [int(x) for x in self.kpt_color[i]] if is_pose else color_palette(i),
                           -1, lineType=cv2.LINE_AA)
            if kpt_line:
                for i, sk in enumerate(self.skeleton):
                    if show_skeleton and sk not in show_skeleton:
                        continue
                    pos1, pos2 = (int(kpts[sk[0]-1][0]), int(kpts[sk[0]-1][1])), (int(kpts[sk[1]-1][0]), int(kpts[sk[1]-1][1]))
                    if (kpts[sk[0]-1][2] < 0.5) or (kpts[sk[1]-1][2] < 0.5):
                        continue
                    cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]],
                             int(line_thickness * plot_size_redio), lineType=cv2.LINE_AA)
            if self.pil:
                self.fromarray(self.im)

    annotator = _Annotator(deepcopy(pose_result.orig_img))
    if pose_result.keypoints is not None:
        for k in reversed(pose_result.keypoints.data):
            annotator.kpts(k, pose_result.orig_shape, kpt_line=True)
    return annotator.result()