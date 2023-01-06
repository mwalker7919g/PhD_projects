"""
Each function represents a routine for the robot to perform on the workflow
YuMi gives positions in terms of 7 coordinates
YuMi-Python interface documentation - https://berkeleyautomation.github.io/yumipy/api/robot.html
"""

from yumipy import YuMiRobot, YuMiState
import time

y = YuMiRobot()

class RobotRoutine():
    def __init__(self, vial_size):
        self.vial_size = vial_size

    def reset_rob_position(vial_size):
        print('reset robot position')

        if vial_size == int(8):

            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-146.06,-122.78,30.53,200.91,126.87,-115.28,54.45]))
        
        elif vial_size == int(20):

            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-146.06,-130.78,30.53,200.91,130.87,-115.28,54.45]))
    
    def robot_rotate(vial_size):
        print('rotating sample for video acquisition')
    
        if vial_size == int(8):

            # left bringtoexchange
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-109.57,-94.07,55.51,159.66,119.65,-92.64,105.22]))

            # l2r exchange
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([50.02,-125.72,17.08,-36.65,69.57,125.83,-86.33]))
            y.right.goto_state(YuMiState([56.01,-125.72,16.01,-44.6,60.94,125.92,-86.06]))
            y.right.close_gripper()
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-62.89,-83.1,75.63,148.78,73.69,-68.21,90.46]))

            y.left.goto_state(YuMiState([-99.14,-77.31,16.63,148.97,-32.33,-75.22,126.03]))
            y.left.close_gripper()
            y.right.open_gripper()
            y.right.set_speed(YuMiRobot.get_v(50))
            y.right.goto_state(YuMiState([50.15,-125.72,10.33,-42.78,75.2,124.36,-79.88]))
            y.right.goto_state(YuMiState([41.8,-125.72,28.3,-43.55,83.6,138.13,-79.14]))
            y.right.goto_state(YuMiState([50.93,-125.72,24.99,-46.83,66.86,132.88,-85.73]))
            y.right.close_gripper()
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-93.43,-66.78,6.56,170.65,-45.14,-101.65,124.74]))

            # rotate
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([40.25,-125.72,5.05,29.13,54.94,98.69,-83.91]))
            y.right.set_speed(YuMiRobot.get_v(700))
            y.right.goto_state(YuMiState([40.2,-125.72,1.56,28.06,56.78,8.73,-83.92]))
            y.right.set_speed(YuMiRobot.get_v(150))
            time.sleep(5)
            
        
        elif vial_size == int(20):

            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79, -143.36, 19.47, 15.26, 42.41, 89.57, -90.64]))
            y.right.set_speed(YuMiRobot.get_v(250)) 
            y.right.goto_state(YuMiState([47.36, -143.36, 16.37, 18.78, 34.78, -6.62, -87.05]))
            y.right.goto_state(YuMiState([47.36, -143.36, 15.41, 19.41, 31.3, -3.28, -82.68]))

    def return_robot_position(vial_size):
        print('returning robot to a central position (for next vial)')

        if vial_size == int(8):
            # right bringtoexchange
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([40.2,-125.72,1.56,28.06,56.78,8.73,-83.92]))
            y.right.goto_state(YuMiState([62.94,-125.72,28.97,-53.41,52.46,139.75,-90.23]))

            # r2l exchange
            y.right.set_speed(YuMiRobot.get_v(50))
            y.left.set_speed(YuMiRobot.get_v(50))
            y.left.close_gripper()
            y.right.open_gripper()
            y.right.goto_state(YuMiState([66.89,-125.72,18.87,-48.45,53.9,133.87,-89.47]))
            y.right.goto_state(YuMiState([70.87,-125.72,15.08,-42.22,61.61,125.84,-88.16]))

            y.right.close_gripper()
            y.left.open_gripper()

            y.left.goto_state(YuMiState([-90.36,-66.75,-14.03,178.46,-64.3,-102.65,122.99]))
            y.left.goto_state(YuMiState([-40.55,-46.81,63.39,177.22,0.51,-115.83,84.29]))
            y.left.goto_state(YuMiState([-55.96,-54.09,71.86,162.38,3.12,-94.54,91.82]))
            y.left.goto_state(YuMiState([-96.38,-54.09,72.25,130.25,104.35,-106.67,93.11]))
            y.left.goto_state(YuMiState([-96.38,-54.09,64.62,146.14,96.75,-97.61,107.57]))
            y.left.close_gripper()
            y.right.open_gripper()
            y.right.goto_state(YuMiState([45.98,-125.72,12.36,-37.03,74.41,120.53,-88.5]))

            # left arm central closed
            y.left.set_speed(YuMiRobot.get_v(50))
            y.left.goto_state(YuMiState([-96.38, -54.09, 64.62, 146.14, 96.75, -97.61, 107.57]))
            y.left.goto_state(YuMiState([-131.55,-54.10,12.1,116.71,97.19,-77.68,95.3]))
        
        elif vial_size == int(20):

            # it then needs to come back to central position
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            # this section is for picking up the vial from a single position:
            # the right arm will drop into a position and the left arm will pick it up
            y.right.set_speed(YuMiRobot.get_v(150))
            y.left.set_speed(YuMiRobot.get_v(150))

            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([77.35, -127.27, 16.54, 14.43, 25.66, 76.58, -88.33]))
            y.right.goto_state(YuMiState([78.42, -127.27, 14.63, 12.96, 25.83, 80.20, -87.12]))
            y.right.open_gripper()
            y.right.goto_state(YuMiState([79.2, -109.81, 10.19, 11.83, 47.74, 88.13, -92.12]))
            y.right.goto_state(YuMiState([37.98, -127.27, 36.2, 10.22, 36.27, 77.22, -113.05]))

            # left arm goes and picks it up, brings it to reasonably central position
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-45.07, -132.64, 31.89, 36.29, 37.6, 39.02, 89.33]))
            y.left.goto_state(YuMiState([-87.69, -132.64, 18.25, 48.35, 24.32, 43.61, 77.82]))
            y.left.goto_state(YuMiState([-91.6, -132.64, 10.3, 47.9, 30.09, 45.08, 73.3]))
            y.left.close_gripper()
            y.left.goto_state(YuMiState([-90.78, -132.64, 10.93, 47.57, 32.87, 43.53, 72.78]))
            y.left.goto_state(YuMiState([-89.3, -132.64, 16.73, 56.53, 25.22, 39.88, 76.11]))
            y.left.goto_state(YuMiState([-50.95, -132.64, 37.28, 52.44, 46.76, 38.01, 85.14]))
    
    def pickup_vial_1(vial_size):
        print('picking vial 1')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-135.31,-125.95,25.11,192.85,113.64,-40.88,62.96]))
            y.left.goto_state(YuMiState([-135.28,-125.95,14.76,192.36,104.76,-37.37,60.46]))
            y.left.close_gripper()
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-135.47,-124.36,17.01,190.06,109.07,-34.65,60.87]))
            y.left.goto_state(YuMiState([-135.49,-124.36,19.04,191.52,110.72,-34.64,62.39]))
            y.left.goto_state(YuMiState([-135.62,-124.35,30.04,191.13,119.18,-34.64,63.66]))
            y.left.goto_state(YuMiState([-137.75,-93.78,37.06,153.87,137.68,-83.6,78.18]))

        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([69.42, -143.36, 25.32, 36.30, 30.15, 54.78, -84.88]))
            y.right.goto_state(YuMiState([82.94, -143.36, 20.79, 36.59, 26.05, 54.78, -71.38]))
            y.right.goto_state(YuMiState([75.8, -122.35, 15.58, 15.48, 27.5, 90.54, -79.01]))
            y.right.goto_state(YuMiState([74.48, -122.35, 11.79, 13.43, 30.29, 90.51, -80.37]))
            y.right.goto_state(YuMiState([79.35, -122.35, 11.35, 14.17, 25.32, 89.37, -77.02]))
            y.right.close_gripper()
            y.right.set_speed(YuMiRobot.get_v(100))
            y.right.goto_state(YuMiState([79.25, -122.35, 11.35, 14.62, 27.32, 88.96, -77.13]))
            y.right.goto_state(YuMiState([79.41, -122.35, 13.55, 13.08, 22.43, 90.11, -76.29]))
            y.right.goto_state(YuMiState([78.05, -122.35, 15.22, 13.09, 24.05, 90.12, -77.04]))
            y.right.goto_state(YuMiState([66.55, -122.35, 24.11, 13.27, 25.5, 87.02, -85.38]))
    
    def pickup_vial_2(vial_size):
        print('picking vial 2')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-134.21,-126.81,22.6,196.64,112.42,-38.59,61.87]))
            y.left.goto_state(YuMiState([-134.2,-126.81,20.81,198.23,111.55,-13.68,62.09]))
            y.left.goto_state(YuMiState([-134.13,-126.81,16.96,196.06,106.8,-7.16,60.11]))
            y.left.close_gripper()
            y.left.goto_state(YuMiState([-134.14,-126.81,18.89,196.14,108.97,-7.19,60.58]))
            y.left.goto_state(YuMiState([-134.14,-126.81,21.87,196.86,111.6,-7.16,61.36]))
            y.left.goto_state(YuMiState([-137.84,-129.56,35.4,201.57,124.1,-11.29,59.32]))

        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([76.15, -143.36, 24.17, 46.16, 30.39, 40.21, -87.67]))
            y.right.goto_state(YuMiState([97.48, -143.36, 20.36, 74.95, 63.61, 8.17, -81.38]))
            y.right.goto_state(YuMiState([97.53, -143.31, 18.3, 73.9, 53.25, 8.31, -76.17]))
            y.right.goto_state(YuMiState([97.53, -143.31, 12.88, 65.26, 50.81, 19.94, -74.56]))
            y.right.close_gripper()
            y.right.set_speed(YuMiRobot.get_v(100))
            y.right.goto_state(YuMiState([96.39, -143.1, 14.32, 64.48, 46.16, 19.89, -74.14]))
            y.right.goto_state(YuMiState([96.07, -143.2, 14.75, 63.83, 46.18, 19.89, -74.56]))
            y.right.goto_state(YuMiState([92.76, -143.1, 19.7, 68.29, 44.58, 14.97, -78.27]))
    
    def pickup_vial_3(vial_size):
        print('picking vial 3')
        if vial_size == int(8):
                
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-138,-122.8,12.79,192.76,103.56,-132.3,61.95]))
            y.left.goto_state(YuMiState([-136.86,-119.74,8.4,190.6,100.27,-133.73,64.48]))
            y.left.close_gripper()
            y.left.goto_state(YuMiState([-136.86,-119.74,10.34,190.74,102.78,-133.72,64.89]))
            y.left.goto_state(YuMiState([-136.87,-119.74,13.07,191.09,104.96,-133.68,65.24]))
            y.left.goto_state(YuMiState([-138.17,-119.74,23.49,193.27,116.27,-133.21,67.56]))
            y.left.goto_state(YuMiState([-122.19,-103.96,57.49,174.32,115.2,-133.22,81.44]))

        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([81.22, -138.94, 21.85, 42.65, 22.92, 52.16, -72.7]))
            y.right.goto_state(YuMiState([73.39, -113.67, 17.35, 18.62, 20.82, 91.54, -79.91]))
            y.right.goto_state(YuMiState([78.01, -113.67, 7.36, 18.69, 31.44, 91.57, -79.9]))
            y.right.goto_state(YuMiState([82.77, -113.67, 5.43, 17.27, 28.99, 92.72, -76.75]))
            y.right.close_gripper()
            y.right.set_speed(YuMiRobot.get_v(100))
            y.right.goto_state(YuMiState([82.56, -113.67, 5.41, 17.43, 30.92, 92.75, -76.79]))
            y.right.goto_state(YuMiState([82.39, -113.67, 5.4, 17.25, 31.75, 92.67, -76.94]))
            y.right.goto_state(YuMiState([82.04, -113.67, 7.81, 17.38, 26.71, 93.25, -76.26]))
            y.right.goto_state(YuMiState([80.54, -113.67, 12.53, 17.45, 21.69, 93.78, -76.2]))
            y.right.goto_state(YuMiState([72.7, -113.67, 21, 17.43, 22.02, 93.78, -79.19]))
    
    def pickup_vial_4(vial_size):
        print('picking vial 4')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(50))
            y.left.goto_state(YuMiState([-134.3,-56.05,13.53,125.48,86.68,-186.27,107.56]))
            y.left.goto_state(YuMiState([-129.64,-122.8,20.43,189.74,104.84,-130.13,61.65]))
            y.left.goto_state(YuMiState([-129.64,-122.79,18.48,186.81,103.99,-130.96,60.18]))
            y.left.close_gripper()
            y.left.set_speed(YuMiRobot.get_v(25))

            y.left.goto_state(YuMiState([-129.64,-122.78,21.43,187.88,107.01,-130.96,61.15]))
            y.left.goto_state(YuMiState([-129.64,-122.76,26.58,182.61,111.17,-130.97,57.63]))
            y.left.goto_state(YuMiState([-135.01,-122.66,37.39,183.84,127.37,-126.49,54.92]))

        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([90.84, -143.36, 22.15, 58.45, 26.6, 25.55, -74.66]))
            y.right.goto_state(YuMiState([107.51, -143.29, 13.24, 70.54, 64.96, 12.57, -69.94]))
            y.right.goto_state(YuMiState([107.61, -143.13, 8.1, 66.51, 62.06, 19.12, -68.33]))
            y.right.goto_state(YuMiState([107.86, -139.08, 7.78, 70.9, 59.16, 18.16, -66.76]))
            y.right.close_gripper()
            y.right.set_speed(YuMiRobot.get_v(100))
            y.right.goto_state(YuMiState([107.75, -139.19, 7.62, 68.99, 58.27, 18.14, -66.69]))
            y.right.goto_state(YuMiState([107.77, -139.18, 7.57, 67.75, 59.31, 17.44, -67.08]))
            y.right.goto_state(YuMiState([107.28, -139.28, 7.98, 66.6, 58.78, 14.77, -68.19]))
            y.right.goto_state(YuMiState([104.4, -139.34, 11.95, 69.13, 59.66, 17.39, -71.11]))
            y.right.goto_state(YuMiState([80.51, -139.36, 31.1, 72.05, 21.84, 10.31, -83.03]))

    def pickup_vial_5(vial_size):
        print('picking vial 5')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-146.06,-122.78,30.53,200.91,126.87,-115.28,54.45]))
            y.left.goto_state(YuMiState([-140.82,-122.76,10.64,191.58,100.36,-127.39,55.31]))
            y.left.goto_state(YuMiState([-142.76,-119.33,3.81,188.47,102.73,-126.84,54.46]))
            y.left.close_gripper()
            y.left.goto_state(YuMiState([-142.76,-119.33,8.32,188.72,106.85,-126.86,55]))
            y.left.goto_state(YuMiState([-142.82,-119.33,25.44,187.78,120.94,-126.86,56.31]))


        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.set_speed(YuMiRobot.get_v(100))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([97.22, -143.33, 20.08, 29.53, 15.17, 58.99, -54.24]))
            y.right.goto_state(YuMiState([84.09, -112.26, 8.41, 36.67, 28.54, 73.82, -76.44]))
            y.right.goto_state(YuMiState([84.53, -112.26, 3.4, 35.22, 33.35, 78.01, -78.37]))
            y.right.goto_state(YuMiState([87.64, -112.26, 1.99, 36.23, 32.06, 75.41, -76]))
            y.right.close_gripper()
            y.right.set_speed(YuMiRobot.get_v(100))
            y.right.goto_state(YuMiState([90.28, -120.13, 6.18, 24.89, 23.95, 75.98, -68.63]))
            y.right.goto_state(YuMiState([88.6, -120.13, 7.75, 28.66, 27.42, 74.58, -70.17]))
            y.right.goto_state(YuMiState([76.68, -132.98, 36.12, 65.66, 26.77, 26.72, -80.38]))
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
    
    def pickup_vial_6(vial_size):
        print('picking vial 6')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-146.06,-122.78,30.53,200.91,126.87,-115.28,54.45]))
            y.left.goto_state(YuMiState([-133.92,-121.13,18.81,190.04,108.61,-124.62,57.11]))
            y.left.goto_state(YuMiState([-133.91,-121.11,13.89,186.22,104.34,-126.51,54.91]))
            y.left.close_gripper()
            y.left.goto_state(YuMiState([-133.92,-121.11,18.94,186.23,109,-126.49,55.1]))
            y.left.goto_state(YuMiState([-132.94,-116.21,29.04,178.88,119.94,-127.36,56.65]))

        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79,-143.36,29.40,29.88,39.61,59.02,-111.76])) # central position
            y.right.goto_state(YuMiState([101.5,-143.36,19.63,60.02,21.97,25.56,-59.81]))
            y.right.goto_state(YuMiState([111.99,-143.33,12.72,67.31,45.49,14.31,-57.19]))
            y.right.goto_state(YuMiState([111.97,-143.32,8.84,61.24,43.97,21.43,-56.03]))
            y.right.goto_state(YuMiState([112.01,-137.83,4.77,62.8,47.26,22.63,-57.45]))
            y.right.close_gripper()
            y.right.set_speed(YuMiRobot.get_v(100))
            y.right.goto_state(YuMiState([111.96,-137.84,4.82,60.17,47.19,23.81,-57.46]))
            y.right.goto_state(YuMiState([111.88,-138,5.06,58.67,47.46,22.56,-57.92]))
            y.right.goto_state(YuMiState([110.7,-139.02,7.81,57.39,42.35,26.7,-56.54]))
            y.right.goto_state(YuMiState([95.64,-141.63,22.63,-0.54,12.92,86.44,-58.65]))
            y.right.goto_state(YuMiState([68.23,-141.63,36.99,2.29,15.34,90.13,-79.16]))
            y.right.set_speed(YuMiRobot.get_v(150))
            y.right.goto_state(YuMiState([41.79,-143.36,29.40,29.88,39.61,59.02,-111.76])) # central position
  
    def pickup_vial_7(vial_size):
        print('picking vial 7')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-146.06,-122.78,30.53,200.91,126.87,-115.28,54.45]))
            y.left.goto_state(YuMiState([-142.87,-116.17,6.03,186.94,104.73,-130.97,55.84]))
            y.left.goto_state(YuMiState([-141.03,-112.97,-0.08,182.61,99.2,-132.24,58.75]))
            y.left.close_gripper()
            y.left.goto_state(YuMiState([-141.04,-112.97,3.52,182.91,102.92,-132.23,59.03]))
            y.left.goto_state(YuMiState([-144.94,-112.97,20.58,182.85,122.8,-133.72,55.28]))


        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([96.16, -143.38, 16.94, -1.05, 6.11, 89.87, -52.6]))
            y.right.goto_state(YuMiState([94.82, -125.97, 9.87, 3.4, 20.26, 98.87, -57.1]))
            y.right.goto_state(YuMiState([88.99, -106.99, -2.35, 17.36, 30.39, 99.39, -71.84]))
            y.right.goto_state(YuMiState([90.97, -106.99, -2.93, 16.53, 27.73, 99.43, -70.69]))
            y.right.close_gripper()
            y.right.goto_state(YuMiState([90.83, -106.99, 1.02, 17.42, 21.76, 98.16, -69.29]))
            y.right.goto_state(YuMiState([89.78, -107.01, 5.6, 24.39, 19.74, 91.39, -69.2]))
            y.right.goto_state(YuMiState([78.3, -107.1, 23.96, 77.81, 38.11, 32.74, -83.99]))
            y.right.goto_state(YuMiState([82.04, -143.38, 31.8, 22.64, 10.33, 68.57, -66.4]))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position

    def pickup_vial_8(vial_size):
        print('picking vial 8')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-146.06,-122.78,30.53,200.91,126.87,-115.28,54.45]))
            y.left.goto_state(YuMiState([-144.18,-126.28,16.12,196.42,107,-109.7,41.33]))
            y.left.goto_state(YuMiState([-144.18,-126.26,11.04,195.41,101.2,-110.77,39.45]))
            y.left.close_gripper()
            y.left.goto_state(YuMiState([-144.18,-126.26,15.85,195.29,105.53,-109.72,40.15]))
            y.left.goto_state(YuMiState([-151.69,-126.16,27.42,204.21,121.03,-111.59,39.25]))

        elif vial_size == int(20):
            y.right.open_gripper()
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position
            y.right.goto_state(YuMiState([105.26,-129.44,8.54,30.59,18.09,60.62,-54.16]))
            y.right.goto_state(YuMiState([105.44,-127.99,3.02,22.94,20.5,70.37,-54.69]))
            y.right.close_gripper()
            y.right.goto_state(YuMiState([105.42,-127.99,5.66,22.53,16.41,70.18,-54.08]))
            y.right.goto_state(YuMiState([103.45,-129.01,11.59,19.03,12.9,74.21,-54.86]))
            y.right.goto_state(YuMiState([76,-129,36.35,17,8.31,79.37,-76.31]))
            y.right.goto_state(YuMiState([41.79, -143.36, 29.40, 29.88, 39.61, 59.02, -111.76]))  # central position

    def putback_vial_1(vial_size):
        print('put back vial 1')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-131.55,-54.1,17.66,121.9,88.45,-83.84,106.4]))
            y.left.goto_state(YuMiState([-138.24,-54.1,7.33,125.61,75.5,-86.9,122.9]))
            y.left.goto_state(YuMiState([-142.37,-54.12,3.72,124.21,75.05,-86.17,127.88]))
            y.left.goto_state(YuMiState([-144.07,-54.33,1.56,123.36,74.27,-87.74,130.46]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-144.06,-54.33,1.56,123.31,74.69,-106.84,130.51]))
            y.left.goto_state(YuMiState([-144.06,-54.32,5.72,121.86,87.55,-101.21,124.35]))

        elif vial_size == int(20):
        
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-79.05, -143.37, 39.48, 181.76, -4.06, -84.3, 81.23]))
            y.left.goto_state(YuMiState([-106.65, -143.37, 23.42, 177.66, -7.37, -80.16, 49.99]))
            y.left.goto_state(YuMiState([-122.89, -142.33, 10.89, 180.21, -13.93, -80.59, 31.49]))
            y.left.goto_state(YuMiState([-122.97, -135.1, 1.76, 178.99, -15.79, -80.57, 31.96]))
            y.left.open_gripper()

    def putback_vial_2(vial_size):
        print('put back vial 2')
        if vial_size == int(8):
            print('small vial 2')
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-131.55,-54.1,17.66,121.9,88.45,-83.84,106.4]))
            y.left.goto_state(YuMiState([-131.51,-54.1,17.09,121.84,93.51,-71.56,107.29]))
            y.left.goto_state(YuMiState([-129.06,-54.1,18.81,125.51,89.56,-82.85,108.4]))
            y.left.goto_state(YuMiState([-137.07,-54.13,7.99,125.96,80.55,-82.23,122.28]))
            y.left.goto_state(YuMiState([-137.06,-54.13,5.81,126.2,74.43,-82.11,124.98]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-132.28,-54.06,15.14,126.68,84.99,-73.48,113.71]))

        elif vial_size == int(20):
            print('big vial 2')
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-70.96, -139.4, 44.02, 179, -4.07, -88.5, 83.38]))
            y.left.goto_state(YuMiState([-99.53, -143.38, 30.49, 186.06, 0.58, -89.56, 63.93]))
            y.left.goto_state(YuMiState([-127.57, -141.28, 12.64, 184.77, -6.83, -82.25, 31.68]))
            y.left.goto_state(YuMiState([-136.62, -139.94, 3.11, 190.71, -13.08, -85.12, 18.49]))
            y.left.open_gripper()

    def putback_vial_3(vial_size):
        print('put back vial 3')
        if vial_size == int(8):
            print('put back small vial 3')
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-134.3,-56.05,13.53,125.48,86.68,-186.27,107.56]))
            y.left.goto_state(YuMiState([-142.99,-56.24,0.12,125.42,76.03,-188.61,123.78]))
            y.left.goto_state(YuMiState([-148.05,-55.55,-3.66,127.02,70.5,-187.67,133.79]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-138.06,-55.43,10.26,127.2,82.95,-182.51,114.09]))

        elif vial_size == int(20):
            print('putback big vial 3')
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-71.89, -117.6, 35.85, 155.62, -8.62, -75.38, 90.03]))
            y.left.goto_state(YuMiState([-96.58, -143.39, 25.07, 177.32, -3.95, -85.28, 58]))
            y.left.goto_state(YuMiState([-116.87, -143.38, 12.28, 210.52, -15.4, -115.77, 33.62]))
            y.left.goto_state(YuMiState([-117.84, -140.87, 8.95, 207.63, -14, -114.15, 33.35]))
            y.left.open_gripper()

    def putback_vial_4(vial_size):
        print('put back vial 4')
        if vial_size == int(8):
            print('put back small vial 4')
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-131.55,-54.1,17.66,121.9,88.45,-83.84,106.4]))
            y.left.goto_state(YuMiState([-121.07,-54.11,22.95,119.64,90.34,-180.42,95.99]))
            y.left.goto_state(YuMiState([-128.07,-54.11,8.81,124.11,74.64,-180.43,113.2]))
            y.left.goto_state(YuMiState([-128.1,-57.01,6.52,127.79,73.93,-179.15,113.36]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-122.27,-56.99,17.81,125.55,85.31,-178.97,101.32]))

        elif vial_size == int(20):
            print('putback big vial 4')
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-55.84, -132.17, 39.58, 182.21, -18.47, -96.25, 96.32]))
            y.left.goto_state(YuMiState([-105.66, -143.37, 23.37, 191.34, -5.42, -95.72, 56.13]))
            y.left.goto_state(YuMiState([-120.52, -140.49, 6.3, 206.14, -20, -108.52, 35.01]))
            y.left.goto_state(YuMiState([-120.53, -139.66, 4.79, 207.4, -19.94, -108.46, 35.25]))
            y.left.open_gripper()

    def putback_vial_5(vial_size):
        print('put back vial 5')
        if vial_size == int(8):
            print('put back small vial 5')
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-134.86,-63.88,11.53,128.64,97.65,-167.81,97.98]))
            y.left.goto_state(YuMiState([-137.93,-63.88,0.8,128.62,90.09,-170.02,104.95]))
            y.left.goto_state(YuMiState([-142.1,-64.86,-9.57,128.24,81.1,-174.6,114.23]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-138.25,-56.84,6.35,119.19,90.46,-177.73,104.96]))

        elif vial_size == int(20):
            print('putback big vial 5')
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-40, -117, 34.37, 172.35, -31.91, -93.18, 104.65]))
            y.left.goto_state(YuMiState([-69.94, -139.06, 34.64, 181.21, -11.71, -88.62, 89.82]))
            y.left.goto_state(YuMiState([-102.68, -143.38, 16.69, 182.07, -15.79, -85.64, 53.36]))
            y.left.goto_state(YuMiState([-110.76, -143.07, 10.74, 180.24, -16.5, -86.85, 43.66]))
            y.left.open_gripper()

    def putback_vial_6(vial_size):
        print('put back vial 6')
        if vial_size == int(8):
            print('put back small vial 6')
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-131.55,-54.1,17.66,121.9,88.45,-83.84,106.4]))
            y.left.goto_state(YuMiState([-125.92,-54.17,12.07,117.86,88.19,-177.39,98.84]))
            y.left.goto_state(YuMiState([-131.49,-54.24,1.81,121,74.21,-177.77,114.43]))
            y.left.goto_state(YuMiState([-134.45,-54.24,-2.08,119.81,71.72,-177.77,119]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-130.47,-33.88,17.38,105.69,69.99,-181.38,115.75]))

        elif vial_size == int(20):
            print('putback big vial 6')
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-41.45, -143.36, 31.81, 207.01, -33.63, -110.32, 97.88]))
            y.left.goto_state(YuMiState([-92.6, -143.36, 19.1, 190.13, -18.61, -93.87, 66.81]))
            y.left.goto_state(YuMiState([-106.22, -138.73, 7, 193.77, -25.84, -99.1, 52.03]))
            y.left.goto_state(YuMiState([-109.87, -135.76, 5.83, 230.54, -27.65, -133.19, 45.96]))
            y.left.open_gripper()

    def putback_vial_7(vial_size):
        print('put back vial 7')
        if vial_size == int(8):
            print('put back small vial 7')
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-131.55,-54.1,17.66,121.9,88.45,-83.84,106.4]))
            y.left.goto_state(YuMiState([-137.62,-50.8,9.1,117.81,88.74,-178.3,104.3]))
            y.left.goto_state(YuMiState([-143.17,-53.65,-2.28,121.81,79.34,-178.27,117.15]))
            y.left.goto_state(YuMiState([-146.48,-55.5,-12.51,116.49,69.31,-173.29,125.23]))
            y.left.goto_state(YuMiState([-146.5,-58.39,-15.63,118.14,68.82,-170.89,124.52]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-138.82,-48.75,7.33,114.25,83.4,-173.44,107.27]))

        elif vial_size == int(20):
            print('putback big vial 7')
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-41.45, -143.36, 31.81, 207.01, -33.63, -110.32, 97.88]))
            y.left.goto_state(YuMiState([-98.15, -143.33, 16.18, 195.34, -20.01, -103.21, 54.93]))
            y.left.goto_state(YuMiState([-102.19, -143.37, 11.8, 201.01, -22.47, -106.84, 50.87]))
            y.left.open_gripper()

    def putback_vial_8(vial_size):
        print('put back vial 8')
        if vial_size == int(8):
            y.left.set_speed(YuMiRobot.get_v(25))
            y.left.goto_state(YuMiState([-131.55,-54.1,17.66,121.9,88.45,-83.84,106.4]))
            y.left.goto_state(YuMiState([-125.82,-54.1,10.31,118.7,88.35,-166.15,96.48]))
            y.left.goto_state(YuMiState([-130.02,-54.1,1.17,117.27,81.39,-170.79,104.9]))
            y.left.goto_state(YuMiState([-133.28,-57.49,-5.87,120.48,75.23,-170.96,111.96]))
            y.left.goto_state(YuMiState([-132.78,-59.25,-8,120.86,73.13,-163.69,111.28]))
            y.left.open_gripper()
            y.left.goto_state(YuMiState([-122.67,-59.23,17.93,126.45,97.01,-166.58,88.27]))

        elif vial_size == int(20):
            print('putback big vial 8')
            y.left.set_speed(YuMiRobot.get_v(150))
            y.left.goto_state(YuMiState([-48.62, -132.68, 43.52, 164.96, -18.83, -88.7, 90.66]))
            y.left.goto_state(YuMiState([-87.24, -138.38, 26.56, 159.06, -8.61, -76.8, 61.17]))
            y.left.goto_state(YuMiState([-107.76, -134.47, 13.07, 162.17, -8.77, -76.8, 37.98]))
            y.left.goto_state(YuMiState([-109.36, -129.74, 6.22, 162.65, -12.01, -76.59, 36.08]))
            y.left.goto_state(YuMiState([-107.8, -126.45, -0.66, 159.13, -16.69, -80.54, 40.05]))
            y.left.open_gripper()