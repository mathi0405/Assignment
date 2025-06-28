#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np

# your LQR implementation in controllers.py must have signature:
# def lqr_controller(x: np.ndarray, x_ref: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray
from controllers import lqr_controller

def odom_cb(msg):
    # extract robot pose
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
    x = np.array([pos.x, pos.y, theta])

    # compute LQR control
    u = lqr_controller(x, X_REF, Q, R)
    v, omega = float(u[0]), float(u[1])

    # publish as Twist
    twist = Twist()
    twist.linear.x = v
    twist.angular.z = omega
    pub.publish(twist)

    rospy.logdebug("LQR → v: %.3f, ω: %.3f", v, omega)

if __name__ == "__main__":
    rospy.init_node("lqr_controller", log_level=rospy.DEBUG)

    # read goal and weights from parameters
    X_REF = np.array([
        rospy.get_param("~x_ref",   0.0),
        rospy.get_param("~y_ref",   1.0),
        rospy.get_param("~yaw_ref", 0.0),
    ], dtype=float)

    Q_diag = rospy.get_param("~Q_diag", [10.0, 10.0, 1.0])
    R_diag = rospy.get_param("~R_diag", [1.0, 1.0])
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)

    rospy.loginfo("LQR Controller starting: goal=(%.2f,%.2f,%.2f), Q=%s, R=%s",
                  X_REF[0], X_REF[1], X_REF[2], Q_diag, R_diag)

    # setup ROS interfaces
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.Subscriber("/odom", Odometry, odom_cb)

    rospy.spin()
