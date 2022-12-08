import copy
import os
import sys

import cv2 as cv
import mediapipe as mp
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.resources import resource_add_path
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.transition import MDSlideTransition
from win32api import GetSystemMetrics

#  to capture screen
from window_capture import WindowCapture as WindowCapture

# annotation 
# from FaceMesh.draw_bounding_rect import calc_bounding_rect , draw_info_text
# from FaceMesh.landmark_list import calc_landmark_list 
# from FaceMesh.pre_process_landmarks import pre_process_landmark , draw_bounding_rect 


# face mesh and utils solutions
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh




# face mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)



width = GetSystemMetrics(0)
height = GetSystemMetrics(1)



cam_on = False


# for splashscreen
class Splash_screen(MDScreen):

    secs = 0
    load = 0
    def __init__(self, *args, **kwargs):
        super(Splash_screen, self).__init__(*args, **kwargs)
        self.orientation = 'vertical'
        Clock.schedule_interval(self.next_window, 15.0/33.0 )

    def next_window(self, *args):
        self.secs = self.secs+1
        self.load += 5.5

        self.ids.loading.value = self.load
        '''  20 seconds'''
        if self.secs == 20:
            self.manager.current = 'home_screen'

    def on_enter(self):
        self.ids.gif.anim_delay = 0.10

# for Home page
class Home_screen(MDScreen ):
# for camera
    def open_camera(self, *args):
       

        self.image = Image()
        self.add_widget(self.image)
        global cam_on
        cam_on = True
        self.capture = cv.VideoCapture(0)

       
        # self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 900)
        # self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 750)
      
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 400)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 400)
        Clock.schedule_interval(self.load_video ,1.0/40.0)


    def load_video(self, *args ):       
        if cam_on:
                self.ret, frame = self.capture.read()
                if self.ret:
                    
                    
                    frame.flags.writeable = False
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    results = face_mesh.process(frame)

                    #Draw the facemesh annotations on the image.
                    frame.flags.writeable = True
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    
                    if results.multi_face_landmarks is not None:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_iris_connections_style())
                                 # Draw the face detection annotations on the image.
                            

                    buffer = cv.flip(frame, 0).tobytes()
                    # buffer = buffer1.tostring()
                    texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                    texture1.blit_buffer(buffer, colorfmt='bgr',bufferfmt = 'ubyte')

                    self.image.texture = texture1
                else:
                    self.ids.no_cam.text = 'NO CAMERA FOUND'



    #button stop
    def close_camera(self, *args):
        global cam_on
        if self.capture:
            cam_on = False
            self.image.opacity = 0
            self.capture.release()

    # go_screen_capture
    def go_to_screen_capture(self , *args):
        self.manager.current = 'screen_capture'
        
        

    #button exit for home 
    def exit_window(self,widget):
        cv.destroyAllWindows()
        exit()


# screen capture screen
class screen_capture(MDScreen):
    def __init__(self, *args, **kwargs):

   
        super().__init__(*args, **kwargs)


    # for screen capture
    def start_screen_capture(self, *args ):
        self.image = Image()
        self.add_widget(self.image)
        Clock.schedule_interval(self.show ,1.0/40.0)
          

    def show(self , *args):

            wincap = WindowCapture()

            screenshot = wincap.get_screenshot()
            self.screen = np.array(screenshot)
            self.screen.flags.writeable = False
            self.screen = cv.cvtColor(self.screen, cv.COLOR_BGR2RGB)
            results = face_mesh.process(self.screen)

            #Draw the facemesh annotations on the image.
            self.screen.flags.writeable = True
            self.screen = cv.cvtColor(self.screen, cv.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=self.screen,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=self.screen,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())


            #texture for kivy
            buff = cv.flip( self.screen, 0).tobytes()
            texture1 = Texture.create(
                size=(self.screen.shape[1], self.screen.shape[0]), 
                colorfmt='bgr')
            texture1.blit_buffer( buff , colorfmt='bgr', bufferfmt = 'ubyte')

            self.image.texture = texture1
            
    # go_to_home button
    def go_to_home(self , *args):
       self.manager.current = 'home_screen'


    # button stop
    def close_screen_capture(self, *args):
        global cam_on
        if cam_on == False:
            self.image.opacity = 0

    #button exit for screen capture
    def exit_window(self,widget):
        cv.destroyAllWindows()
        exit()



class Application(MDApp):

    def build(self):
        self.icon = 'assets/icons/icon.png'
        self.title = "Student Emotion Recognition"
        # theme/s of the app
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Light"
    

        sm = MDScreenManager(transition= MDSlideTransition())
        sm.add_widget(Splash_screen(name ='splash_screen'))
        sm.add_widget(Home_screen(name='home_screen'))
        sm.add_widget(screen_capture(name= "screen_capture"))
        return sm


if __name__ == '__main__':
    #this method is  to find the error of the code
        try:
            if hasattr(sys, '_MEIPASS'):
                resource_add_path(os.path.join(sys._MEIPASS))
            app = Application()
            app.run()
        except Exception as e:
            print(e)
            input("Press enter to exit.")
