 #!/usr/bin/env python3
 import os, sys
 sys.path.insert(0, os.path.dirname(__file__))

 import rospy
 from geometry_msgs.msg import Twist
 from nav_msgs.msg       import Odometry
+from tf.transformations import euler_from_quaternion
 import numpy as np
 from controllers        import mpc_controller

 def odom_cb(msg):
     # extract position & yaw
     p = msg.pose.pose.position
     o = msg.pose.pose.orientation
-    _, _, θ = tf.transformations.euler_from_quaternion([o.x,o.y,o.z,o.w])
+    _, _, θ = euler_from_quaternion([o.x, o.y, o.z, o.w])
     x = np.array([p.x, p.y, θ])

     # compute MPC sequence
-    u_seq = mpc_controller(x, X_REF, A, B, Q, R, N)
+    u_seq = mpc_controller(x, X_REF, A, B, Q, R, N)
     v, ω = u_seq[0]

-   twist = Twist()
+    twist = Twist()
     twist.linear.x  = float(v)
     twist.angular.z = float(ω)
     pub.publish(twist)

 if __name__=="__main__":
     rospy.init_node("mpc_controller")

     # parameters
     X_REF = np.array([
         rospy.get_param("~x_ref",   0.0),
         rospy.get_param("~y_ref",   1.0),
         rospy.get_param("~yaw_ref", 0.0),
     ])
     A = np.array(rospy.get_param("~A", [[1,0,0],[0,1,0],[0,0,1]]))
     B = np.array(rospy.get_param("~B", [[0.1,0],[0,0.1],[0,0.0]]))
     Q = np.diag( rospy.get_param("~Q_diag", [10.0,10.0,1.0]) )
     R = np.diag( rospy.get_param("~R_diag", [1.0,1.0]) )
     N = rospy.get_param("~N", 15)

-    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
-    rospy.Subscriber("/odom", Odometry, odom_cb)
+    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
+    rospy.Subscriber("/odom", Odometry, odom_cb)

     rospy.loginfo("MPC controller started with horizon %d", N)
     rospy.spin()
