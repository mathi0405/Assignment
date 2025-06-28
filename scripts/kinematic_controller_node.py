#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg       import Odometry
import tf
import math

def wrap_pi(angle):
    """Normalize an angle to [-π, π]."""
    return (angle + math.pi) % (2*math.pi) - math.pi

def odom_cb(msg):
    # 1) Current pose
    px = msg.pose.pose.position.x
    py = msg.pose.pose.position.y
    q  = msg.pose.pose.orientation
    _, _, theta = tf.transformations.euler_from_quaternion(
        [q.x, q.y, q.z, q.w]
    )

    # 2) Goal and gains (unpack for clarity)
    xg, yg, phig = X_REF
    k_rho, k_alpha, k_beta = GAINS

    # 3) Compute errors
    dx  = xg - px
    dy  = yg - py
    rho = math.hypot(dx, dy)                             # distance error
    alpha = wrap_pi(math.atan2(dy, dx) - theta)          # heading error to line-of-sight
    beta  = wrap_pi(phig - math.atan2(dy, dx))           # final orientation error

    # 4) Basic “go-to-goal” law
    v_raw     = k_rho   * rho
    omega_raw = k_alpha * alpha + k_beta * beta

    # 5) Prevent circling: only drive when roughly facing the goal
    v_adj = v_raw * math.cos(alpha)  # zero when α ≈ ±90°

    # 6) Two-phase logic: drive-then-spin
    if rho > RHO_THRESH:
        # Phase 1: move toward the point
        v_cmd     = v_adj
        omega_cmd = omega_raw
    else:
        # Phase 2: at the point, stop and correct yaw
        v_cmd = 0.0
        yaw_err = wrap_pi(phig - theta)
        if abs(yaw_err) > YAW_THRESH:
            omega_cmd = K_YAW * yaw_err
        else:
            # goal fully reached
            omega_cmd = 0.0

    # 7) Saturate to safe speeds
    v_cmd     = max(-V_MAX,    min(V_MAX,    v_cmd))
    omega_cmd = max(-OMEGA_MAX, min(OMEGA_MAX, omega_cmd))

    # 8) Log for visibility
    rospy.loginfo(
        "ρ=%.3f  α=%.3f  →  v=%.3f  ω=%.3f",
        rho, alpha, v_cmd, omega_cmd
    )

    # 9) Publish
    twist = Twist()
    twist.linear.x  = v_cmd
    twist.angular.z = omega_cmd
    pub.publish(twist)

if __name__ == "__main__":
    rospy.init_node("kinematic_controller")

    # Read goal & gains (with defaults)
    X_REF = (
        rospy.get_param("~x_ref",   0.0),
        rospy.get_param("~y_ref",   1.0),
        rospy.get_param("~yaw_ref", math.pi),
    )
    GAINS = (
        rospy.get_param("~k_rho",   0.5),
        rospy.get_param("~k_alpha", 1.5),
        rospy.get_param("~k_beta", -0.3),
    )
    rospy.loginfo("Goal = %s   Gains = %s", X_REF, GAINS)

    # Read thresholds & max speeds
    RHO_THRESH = rospy.get_param("~rho_thresh", 0.05)   # [m]
    YAW_THRESH = rospy.get_param("~yaw_thresh", 0.05)   # [rad]
    K_YAW      = rospy.get_param("~k_yaw",       1.0)    # gain for yaw-only phase
    V_MAX      = rospy.get_param("~v_max",       0.5)    # [m/s]
    OMEGA_MAX  = rospy.get_param("~omega_max",   1.0)    # [rad/s]

    # Publisher & subscriber
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.Subscriber("/odom", Odometry, odom_cb)

    rospy.spin()
