import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from screeninfo import get_monitors
from multiprocessing import Process, Pipe
import cv2


class DisplayImage:
    def __init__(self, process):
        self.normalizer = mpl.colors.Normalize(vmin=0, vmax=0.5)
        self.mapper = cm.ScalarMappable(norm=self.normalizer, cmap='magma')
        self.alpha = 0.2
        self.beta = 1 - self.alpha
        self.screen_size = get_monitors()[0]  # Assumes displaying on first monitor available
        self.process = process
        if self.process:
            child_conn, self.parent_conn = Pipe()
            Process(target=self.show_image_in_separate_process, args=(child_conn,)).start()

    def display(self, original_img, disparity, fps, original_width, original_height, blended):
        if self.process:
            try:
                self.parent_conn.send((original_img, disparity, fps, original_width, original_height, blended))
            except:
                pass
        else:
            self.show_image(original_img, disparity, fps, original_width, original_height, blended)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close()

    def show_image_in_separate_process(self, conn):
        while True:
            original_img, disparity, fps, original_width, original_height, blended = conn.recv()
            self.show_image(original_img, disparity, fps, original_width, original_height, blended)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close()
                break

    def show_image(self, original_img, disparity, fps, original_width, original_height, blended):
        # Format disparity image
        colormapped_im = (self.mapper.to_rgba(disparity)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        result_img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

        # Calculate dimensions for displaying each window
        new_width = round(self.screen_size.width / 3 - 5) if blended else round(self.screen_size.width / 2 - 240)
        new_height = round(original_height * new_width / original_width)
        new_dims = (new_width, new_height)

        # Get images into a tuple and resize them
        if blended:
            blended_result = cv2.addWeighted(original_img, self.alpha, result_img, self.beta, 0.0)
            cv2.putText(blended_result, str(fps), (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
            images = (cv2.resize(original_img, dsize=new_dims), cv2.resize(result_img, dsize=new_dims),
                      cv2.resize(blended_result, dsize=new_dims))
        else:
            cv2.putText(result_img, str(fps), (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
            images = (cv2.resize(original_img, dsize=new_dims), cv2.resize(result_img, dsize=new_dims))

        # Concatenate and display images in one window
        all_images = np.concatenate(images, axis=1)
        cv2.imshow('Results', all_images)

    def close(self):
        print('-> Done')
        print('   Closing windows')
        cv2.destroyAllWindows()
