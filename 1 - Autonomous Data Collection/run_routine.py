from records_vids_classes import *
from robot_positions import *
from yumipy import YuMiRobot, YuMiState
import time

if __name__ == "__main__":
    vial_size_input = int(input("Vial size = "))
    vial = RobotRoutine()
    
    for i in range(1,9):

            vial.reset_rob_position(vial_size_input)
            pick_vial_function = str("pickup_vial_"+str(i)+"()")
            eval(vial.pick_vial_function(vial_size_input))
            vial.robot_rotate(vial_size_input)
            start_video_recording('video_vial_'+str(i)+'.avi', vial_size_input)
            if vial_size_input == int(8):
                time.sleep(5)
            elif vial_size_input == int(20):
                time.sleep(20)
            stop_video_recording('video_vial_'+str(i)+'.avi')
            vial.return_robot_position(vial_size_input)
            putback_vial = str("putback_vial_"+str(i)+"()")
            eval(vial.putback_vial(vial_size_input))   

    else:
        print("haven't done routine with this vial size")