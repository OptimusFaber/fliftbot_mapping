import numpy as np
import copy
from threading import Lock
from fliftbot_utils.utils import getFliftbotConfigsPath, getFliftbotTmpsPath, SIMULATOR_CONFIG_FILENAME
import fliftbot_utils.namespace as namesp
import random


class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

def getWhlPoint(rot_c : Point, angle, length) -> Point:
    return Point(rot_c.x + length * np.cos(np.radians(angle)), rot_c.y + length * np.sin(np.radians(angle)))

def unitVector(vector_coords):
    """ Returns the unit vector of the vector.  """
    return vector_coords / np.linalg.norm(vector_coords)

def angleBtwVectors(a, b, module=True, degrees=True): # TODO: what if anti-collinear and continuous angle?
    """ Returns min angle btw 2d-vectors w/ sign
        or 0 in case of any zero-vector:
        >>> angleBtwVectors([2,0], [2,-2])
        -45.00000000000001
        >>> angleBtwVectors([1,0], [-1,-1])
        -135.0
        >>> angleBtwVectors([34,34], [34,34])
        0.0
        >>> angleBtwVectors([0,0], [1,-1])
        0.0
    """
    if (np.linalg.norm(a) == 0. or np.linalg.norm(b) == 0.):
        return 0.
    
    X = 0
    Y = 1
    k = (180. / np.pi) if degrees else 1
    a_u = unitVector(a)
    b_u = unitVector(b)
    sign = np.sign(a[X]*b[Y] - a[Y]*b[X])
    if sign == 0:
        sign = 1.0
    
    result = sign * np.arccos(np.clip(np.dot(a_u, b_u), -1.0, 1.0)) * k
    return result % 360 if module else result

class LidarSim:
    def __init__(self):
        self.enabled = False
        self.r_min_cm = 0 # meters
        self.r_max_cm = 8 * 100 # meters
        self.r_dense_cm = 10
        self.points : list[float] = []
        self.points_atomic = []
        self.lock = Lock()

        self.obstacle_directions : dict[float, float] = {}

    def detect_obstacle_(self, angle_dgr, dist):
        # key = round(angle_dgr, 0) % 360
        # round +- 5
        key = (round(angle_dgr * 2, -1) / 2) % 360
        if key in self.obstacle_directions.keys():
            self.obstacle_directions[key] = min(dist, self.obstacle_directions[key]) 
        else:
            self.obstacle_directions[key] = dist

    # TODO: check!
    def fastScan(self, pos : Point, orientation_dgr : float, obstacles : list[Point]):
        if not self.enabled:
            return
        self.points.clear()
        tmp = []
        self.obstacle_directions.clear()
        v0 = [1, 0]
        for p in obstacles:
            v1 = [p.x - pos.x, p.y - pos.y]
            norm = np.linalg.norm(v1)
            if not ((0.01 * self.r_min_cm) <= norm <= (0.01 * self.r_max_cm)):
                continue

            # detection angle is in lidar basis!
            

            angle_dgr = (0.0 if norm == 0.0 or norm == np.nan else angleBtwVectors(v0, v1)) - orientation_dgr
            self.detect_obstacle_(angle_dgr, norm)
        
        for a, r in self.obstacle_directions.items():
                tmp += [r * np.cos(np.radians(a+90)), r * np.sin(np.radians(a+90))]
            
        self.points += tmp
        self.lock.acquire()
        try:
            self.points_atomic = copy.deepcopy(self.points)
        finally:
            self.lock.release()
        tmp.clear()
        return
        
        for l, p1, p2 in self.walls:
            
            # get intersections in global coords
            intersections = cornerPoints(np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), self.r_max_cm * 0.01, np.array([pos.x, pos.y]))
            # no intersections
            if (intersections is None):
                continue

            c1, c2 = intersections
            X = 0
            Y = 1
            
            # get angle range in global basis orientation but relative to robot pos: from 0 to 360 degrees
            
            v1 = [c1[X] - pos.x, c1[Y] - pos.y]
            v2 = [c2[X] - pos.x, c2[Y] - pos.y]
            start_dgr = angleBtwVectors(v0, v1, module=False)
            end_dgr = angleBtwVectors(v0, v2, module=False)
            if start_dgr > end_dgr:
                start_dgr, end_dgr = end_dgr, start_dgr

            # 
            # if not (0 <= end_dgr - start_dgr <= 180):
            #     print(start_dgr, end_dgr, file=sys.stderr)
            
            
            for angle_dgr in np.arange(start_dgr, end_dgr, 1):
                x = np.cos(np.radians(angle_dgr)) * self.r_max_cm * 0.01 + pos.x # in global basis
                y = l(x)
                norm = np.linalg.norm([x - pos.x, y - pos.y])
                if (not np.isnan(y) and norm <= (self.r_max_cm * 0.01)):
                    angle_dgr = angleBtwVectors(v0, [x - pos.x, y - pos.y])
                    # detection angle is in lidar basis!
                    self.detect_obstacle_(angle_dgr - orientation_dgr, norm)
            
            for a, r in self.obstacle_directions.items():
                tmp += [r * np.cos(np.radians(a+90)), r * np.sin(np.radians(a+90))]
            
            self.points += tmp
            self.points_atomic = self.points
                
            

    
    def scan(self, pos : Point, orientation_dgr : float):
        return
        self.points.clear()
        for a in [val for val in range(0, 360, 5)]:
            for r in [val / 100 for val in range(self.r_min_cm, self.r_max_cm, self.r_dense_cm)]:
                collision = False
                
                # process points
                for p in self.obstacles:
                    x = pos.x + r * np.cos(np.radians(orientation_dgr + a))
                    y = pos.y + r * np.sin(np.radians(orientation_dgr + a))
                    if (np.linalg.norm([x - p.x, y - p.y])) <= (self.r_dense_cm / 100.):
                        # self.points += [(p.x - pos.x)*np.cos(np.radians(orientation_dgr)), (p.y - pos.y)*np.sin(np.radians(orientation_dgr))]
                        self.points += [r * np.cos(np.radians(a+90)), r * np.sin(np.radians(a+90))] # 90 because lidar is rotated on 90 degrees in Oxy
                        self.points_atomic = self.points
                        collision = True
                        break
                
                if collision:
                    break

                
                
                # process lines
                for f, i1, i2 in self.walls:
                    x = pos.x + r * np.cos(np.radians(orientation_dgr + a))
                    y = pos.y + r * np.sin(np.radians(orientation_dgr + a))
                    # print(f(1), f(2), f(3), file=sys.stderr)
                    if (np.isnan(f(x))):
                        continue
                    
                    if (np.linalg.norm([0, y - f(x)])) <= (self.r_dense_cm / 100.):
                        # self.points += [(x - pos.x)*np.cos(np.radians(orientation_dgr)), (y - pos.x)*np.sin(np.radians(orientation_dgr))]
                        self.points += [r * np.cos(np.radians(a+90)), r * np.sin(np.radians(a+90))] # 90 because lidar is rotated on 90 degrees in Oxy
                        self.points_atomic = self.points
                        collision = True
                        break

                if collision:
                    break
        
        # self.lock.acquire()
        # try:
        self.points_atomic = self.points
        # finally:
        #     self.lock.release()
    
    def data(self) -> list | None:
        if not self.enabled:
            return None
        self.lock.acquire()
        try:
            tmp = self.points_atomic
            # print(tmp, file=sys.stderr)
        finally:
            self.lock.release()
        return tmp


# class LiftSrv(Node):
#     def __init__(self, namespace=nsp.NAMESPACE):
#         """ROS"""
#         super().__init__(f'{namespace}_lift_service', namespace=namespace)
#         if (f"/{namespace}" != self.get_namespace()):
#             raise RuntimeError(f"User namespace /{namespace} was overwritten by ros: {self.get_namespace()}") 
        
#         self.lift_srv = self.create_service(Lift, 'lift', self.execute_callback)

#     def execute_callback(self, request, response):
#         self.get_logger().info('Executing goal...')
#         time.sleep(2)
#         response.result = 'Success'
#         return response

class RobotSim:
    def __init__(self, id = 0):
        self.id = id
        self.lidar = LidarSim() 
        self.mass = 1
        self.len = 1.5
        self.body_len = 1.5
        self.acceleration = 1
        self.down_acceleration = 1
        self.whl_angular_velocity = 20.0
        self.velocity = 0.0
        self.prev_values = np.array([])
        self.t0 = 0.0
        self.t = 0.0
        self.global_time = 0.0
        self.cmd_delay = 1.0
        self.last_cmd_t = 0.0
        self.vel_limit = 50
        self.target_velocity = 0.0

        self.rot_c = Point()
        self.angle = 0.0 # degrees
        self.whl_angle = 0.0
        self.target_whl_angle = 0.0
        self.whl_c = getWhlPoint(self.rot_c, self.angle, self.len)
        self.front_c = getWhlPoint(self.rot_c, self.angle, self.body_len)

        self.angle_offset_dgr = 0.0

        self.stop_p1p2p3p4_xy = [0, 0, 0, 0, 0, 0, 0, 0]
        self.warn_p1p2p3p4_xy = [0, 0, 0, 0, 0, 0, 0, 0]
        self.outer_circle = (0.0, 0.0, 1.0)
        self.inner_circle = (0.0, 0.0, 1,0)
        self.lidar_semantic_zone_p1p2p3p4_xy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.semantic_lidar_range = 5.0

        # metadata for scheduler
        self.cur_point = None
        self.next_point = None
        self.next_point2 = None
        self.prev_point = None

        self.velocity_raw = False
    
    ''' Getters '''
    def getOdom(self, cyclic=False):
        return self.rot_c.x, self.rot_c.y, self.angle if not cyclic else self.angle % 360
    

    ''' Setters '''
    def set_wheel_base_len_m(self, len=1.5):
        assert(len > 0)
        self.len = float(len)
        self.whl_c = getWhlPoint(self.rot_c, self.angle, self.len)

    def set_body_len_m(self, len=1.5): # used for visualization & lidars
        assert(len > 0)
        self.body_len = float(len)
        self.front_c = getWhlPoint(self.rot_c, self.angle, self.body_len)

    def set_angle_offset(self, angle_dgr : float):
        assert abs(angle_dgr) <= 90.0
        self.angle_offset_dgr = angle_dgr 
    def setAcceleration(self, acceleration):
        assert acceleration >= 0.
        self.acceleration = acceleration
    
    def setWhlAngle(self, angle):
        self.whl_angle = angle + self.angle_offset_dgr
        self.target_whl_angle = angle + self.angle_offset_dgr

    def setVelocity(self, velocity):
        # in our system negative torque means ==[] --> move, but geometrically it must be positive velocity cause ==[] w/ yaw=0 is collinear Ox
        self.velocity = -velocity
        self.target_velocity = -velocity
    
    def setVelocityLimit(self, v):
        self.vel_limit = v

    def setPosition(self, x, y, angle):
        self.angle = angle
        self.angle0 = angle
        
        self.rot_c = Point(x, y)
        self.rot_c0 = Point(x, y)
        
        self.whl_c = getWhlPoint(self.rot_c, self.angle, self.len)
        self.front_c = getWhlPoint(self.rot_c, self.angle, self.body_len)
    
    def setPositionWithoutAngle(self, x, y):
        
        self.rot_c = Point(x, y)
        self.rot_c0 = Point(x, y)
        
        self.whl_c = getWhlPoint(self.rot_c, self.angle, self.len)
        self.front_c = getWhlPoint(self.rot_c, self.angle, self.body_len)
    
    def moveRotCenter(self, velocity, angle):
        dt = self.t - self.t0
        # print(self.rot_c.x, self.rot_c0.x)
        self.v = min(np.linalg.norm([self.rot_c.x - self.rot_c0.x, self.rot_c.y - self.rot_c0.y]) / dt, velocity)
        # print(self.v)
        acceleration = self.acceleration if self.v < velocity else 0
        
        self.angle0 = self.angle
        self.rot_c0 = copy.deepcopy(self.rot_c)
        
        self.angle = angle
        self.rot_c.x += (self.v * dt + acceleration * dt**2 / 2) * np.cos(np.radians(self.angle))
        self.rot_c.y += (self.v * dt + acceleration * dt**2 / 2) * np.sin(np.radians(self.angle))
        self.t0 += dt
        self.whl_c = getWhlPoint(self.rot_c, self.angle, self.len)
        self.front_c = getWhlPoint(self.rot_c, self.angle, self.body_len)
    
    def setDownAcceleration(self, value):
        self.down_acceleration = abs(value)
    
    def setWhlAngularVel(self, value):
        '''
        Set angular velocity in degrees per second
        '''
        self.whl_angular_velocity = abs(value)

    def dump_feedback(self):
        with open(f"{getFliftbotTmpsPath()}/{namesp.NAMESPACE_PREFIX}{self.id}_sim_feedback.txt", "w") as f:
            f.write(f"{self.velocity:4.2f}, {self.whl_angle:4.2f}\n")
    
    def moveBycycle(self):
        dt = self.t - self.t0
        self.global_time += dt
        if self.global_time > self.last_cmd_t and (not self.velocity_raw):
            self.target_velocity = 0.0

        divisor = self.whl_angular_velocity * dt
        if abs(divisor) <= 0.00001:
            divisor = 0.00001
        tmp_val = self.whl_angular_velocity * dt * np.sign(self.target_whl_angle - self.whl_angle) * np.clip(abs(self.target_whl_angle - self.whl_angle) / (divisor), 0, 1)
        if (np.isnan(tmp_val)):
            tmp_val = 0.0
        if abs(self.whl_angular_velocity) > 100:
            self.whl_angle = self.target_whl_angle
        else:
            self.whl_angle += tmp_val 

        tmp = self.acceleration if np.sign(self.target_velocity) == np.sign(self.velocity) else self.down_acceleration
        acceleration = tmp * np.sign(self.target_velocity - self.velocity)
        
        if self.velocity_raw:
            self.velocity = self.target_velocity
        else:
            if abs(acceleration) > 100:
                self.velocity = np.clip(self.target_velocity, -abs(self.vel_limit), abs(self.vel_limit))
            else:
                self.velocity = np.clip(self.velocity + acceleration * dt, -abs(self.vel_limit), abs(self.vel_limit))
        self.rot_c.x += self.velocity * np.cos(np.radians(self.angle)) * dt * np.cos(np.radians(self.whl_angle))
        self.rot_c.y += self.velocity * np.sin(np.radians(self.angle)) * dt * np.cos(np.radians(self.whl_angle))
        dw = np.degrees(self.velocity * np.sin(np.radians(self.whl_angle)) / self.len) * dt
        self.angle += dw
        # if round(self.global_time*10) % 50 == 0:
        #     self.angle += random.uniform(0, 360)
        self.velocity *= 1
        self.whl_c = getWhlPoint(self.rot_c, self.angle, self.len)
        self.front_c = getWhlPoint(self.rot_c, self.angle, self.body_len)

    def reachVelocity(self, velocity):
        # in our system negative torque means ==[] --> move, but geometrically it must be positive velocity cause ==[] w/ yaw=0 is collinear Ox
        self.target_velocity = -velocity
        self.last_cmd_t = self.global_time + self.cmd_delay
    
    def reachWhlAngle(self, angle):
        self.target_whl_angle = angle + self.angle_offset_dgr
        self.last_cmd_t = self.global_time + self.cmd_delay

    def resetTime(self):
        self.t = 0.0
        self.t0 = 0.0
        self.last_cmd_t = 0.0
        self.global_time = 0.0
    
    def updTime(self, dt):
        self.t0 = self.t
        self.t += dt