from roboschool.scene_abstract import Scene, cpp_household
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.gym_forward_walkers import RoboschoolHumanoid
from roboschool.gym_forward_walkers import RoboschoolForwardWalkersBase
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys

class TennisScene(Scene):
    multiplayer = False
    players_count = 1
    VIDEO_W = 600
    VIDEO_H = 400

    def __init__(self):
        Scene.__init__(self, gravity=9.8, timestep=0.0165/4, frame_skip=4)
        self.score_left = 0
        self.score_right = 0

    def actor_introduce(self, robot):
        i = robot.player_n - 1

    def episode_restart(self):
        Scene.episode_restart(self)
        self.ground_plane_mjcf = self.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "mujoco_assets/ground_plane.xml"))
        fpose = cpp_household.Pose()
        self.field = self.cpp_world.load_thingy(
            os.path.join(os.path.dirname(__file__), "models_outdoor/stadium/tennis1.obj"),
            fpose, 1.0, 0, 0xFFFFFF, True)
        fpose.set_xyz(0,0,0.3)
        self.tennis_net_urdf = self.cpp_world.load_urdf(os.path.join(os.path.dirname(__file__), "models_outdoor/stadium/tennis_net.urdf"), fpose, True)
        if self.score_right + self.score_left > 0:
            sys.stdout.write("%i:%i " % (self.score_left, self.score_right))
            sys.stdout.flush()
        self.camera = self.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        self.camera_itertia = 0
        self.frame = 0
        self.score_left = 0
        self.score_right = 0
        self.restart_from_center(self.players_count==1 or self.np_random.randint(2)==0)

    def restart_from_center(self, leftwards):
        # self.np_random.uniform(low=2.0, high=2.5) * (-1 if leftwards else +1))
        self.timeout = self.TIMEOUT
        self.timeout_dir = (-1 if leftwards else +1)
        self.bounce_n = 0

    def global_step(self):
        self.frame += 1

        if not self.multiplayer:
            # Trainer
            pass

        Scene.global_step(self)

    TIMEOUT = 150

    def HUD(self, a, s):
        self.cpp_world.test_window_history_advance()
        self.cpp_world.test_window_observations(s.tolist())
        self.cpp_world.test_window_actions(a.tolist())
        s = "%04i TIMEOUT%3i %0.2f:%0.2f" % (
            self.frame, self.timeout, self.score_left, self.score_right
            )
        self.cpp_world.test_window_score(s)
        self.camera.test_window_score(s)

    def camera_adjust(self):
        self.camera_itertia *= 0.9
        self.camera_itertia += 0.1 * 0.05*self.ball_x
        self.camera.move_and_look_at(0,-1.0,1.5, self.camera_itertia,-0.1  ,0)

class TennisSceneMultiplayer(TennisScene):
    multiplayer = True
    players_count = 2


# -- Environment itself here --

class RoboschoolTennis(RoboschoolHumanoid):
    #random_lean = True
    foot_list = ["right_foot", "left_foot", "racquet_head"]  # "left_hand"

    def __init__(self):
        RoboschoolHumanoid.__init__(self, model_xml="../models_robot/tennis_humanoid.xml")
        high = np.ones([17+3])  # three more joints
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([44+1+6+9])  # 1 more foot, 6: pos+speed additional joints, 9: command
        self.observation_space = gym.spaces.Box(-high, high)
        self.electricity_cost /= 4
        self.training_hit_ball_counter = 0
        self.training_walk_counter = 0

    def create_single_player_scene(self):
        self.player_n = 0
        s = TennisScene()
        s.np_random = self.np_random
        return s

    TENNIS_HALFLEN    = 23.77/2*0.5   # see random_stadium.py
    TENNIS_HALFWIDTH  =  8.23/2*0.5
    STARTPOS_X        = TENNIS_HALFLEN+0.5
    STARTPOS_Y        = -0.8
    TYPICAL_BALL_SPEED = 2.0

    def robot_specific_reset(self):
        RoboschoolForwardWalkersBase.robot_specific_reset(self)  # not RoboschoolHumanoid
        self.motor_names  = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power  = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_tilt", "right_elbow_bend"]
        self.motor_power += [75, 75, 30, 50]
        self.motor_names += ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_tilt", "left_elbow_bend"]
        self.motor_power += [75, 75, 30, 50]
        self.motor_names += ["racquet_rock"]
        self.motor_power += [20]
        self.motors = [self.jdict[n] for n in self.motor_names]
        self.racquet = self.parts["racquet_head"]
        self.flag_reposition()
        self.random_yaw = not self.training_hit_ball
        self.random_orientation(False)
        self.move_robot(self.STARTPOS_X, self.STARTPOS_Y, 0)

    def flag_reposition(self):
        self.walk_target_x = -self.TENNIS_HALFLEN/2
        self.walk_target_y = 0

        self.command_hitvector_sign = +1 if self.np_random.randint(2)==0 else -1
        self.command_hitvector_whitened = self.np_random.uniform(low=-1, high=+1, size=(3,))
        self.command_hitvector = np.array([
            -(self.command_hitvector_whitened[0]*0.5 + 1.0)*self.TYPICAL_BALL_SPEED,
             (self.command_hitvector_whitened[0]*0.3      )*self.TYPICAL_BALL_SPEED,
             (self.command_hitvector_whitened[0]*0.3 + 0.2)*self.TYPICAL_BALL_SPEED
            ])
        hit_z  = self.np_random.uniform(low=0.1, high=(2.3 if self.command_hitvector_sign==+1 else 1.5))
        hit_dx = -0.5 if self.command_hitvector_sign==+1 else 0  # hit on the left should be a bit forward

        self.training_hit_ball = self.training_hit_ball_counter < self.training_walk_counter  # equal number of timesteps to hit and walk
        if self.training_hit_ball:
            self.command_body_x = self.STARTPOS_X
            self.command_body_y = self.STARTPOS_Y
            self.command_hitpoint = np.array([
                self.command_body_x + hit_dx,
                self.command_body_y - self.command_hitvector_sign*self.np_random.uniform(low=0.5, high=+0.8) + 0.2,
                hit_z
                ])
        else:
            # walk + hit
            self.command_hitpoint = np.array([  # anywhere
                self.np_random.uniform(low=+self.TENNIS_HALFLEN,   high=+self.TENNIS_HALFLEN*0.3),
                self.np_random.uniform(low=-self.TENNIS_HALFWIDTH, high=+self.TENNIS_HALFWIDTH),
                hit_z
                ])
            self.command_body_x = self.command_hitpoint[0] - hit_dx  # body pos based on hit pos
            self.command_body_y = self.command_hitpoint[1] + self.command_hitvector_sign*self.np_random.uniform(low=0.5, high=+0.8) - 0.2  # 0.2 because racquet in the right hand

        self.command_hitpoint_debugsphere = self.scene.cpp_world.debug_sphere(*self.command_hitpoint, 0.05, 0xFF8080)
        # place body based on ball point
        self.command_body_debugsphere = self.scene.cpp_world.debug_sphere(self.command_body_x, self.command_body_y, 0.1, 0.1, 0xFF8080)
        self.trainer_timeout = 50 if self.training_hit_ball else 150
        self.command_hit_importance_integral = 0.0
        self.reward_hit_pos_integral = 0.0
        self.reward_hit_speed_integral = 0.0
        self.reward_hit_orientation_integral = 0.0
        self.on_ground_frame_counter = 0
        self.crawl_start_potential = None
        self.crawl_ignored_potential = 0.0

    def calc_state(self):
        state = RoboschoolHumanoid.calc_state(self)

        self.trainer_timeout -= 1
        reset_potential = False
        z = self.body_xyz[2]
        if self.trainer_timeout <= 0 and z > 0.8:
            self.flag_reposition()
            state = RoboschoolHumanoid.calc_state(self)  # calculate state again, against new flag pos
            reset_potential = True

        self.racquet_pose = self.racquet.pose()
        self.racquet_xyz  = np.array(self.racquet_pose.xyz())
        self.racquet_v    = self.racquet.speed()
        e_pose = cpp_household.Pose()
        e_pose.set_xyz(1, 0, 0)
        racquet_face = self.racquet_pose.dot(e_pose)
        racquet_face_vector  = np.array(racquet_face.xyz()) - self.racquet_xyz
        racquet_face_vector *= -self.command_hitvector_sign
        # self.racquet_v and racquet_face_vector should point in the same direction
        correct_orientation = self.command_hitvector / np.linalg.norm(self.command_hitvector)
        how_good_is_orientation = 1 - 0.5*np.linalg.norm(racquet_face_vector - correct_orientation)  # 0..1

        if self.scene.human_render_detected:
            self.racquet_face_debug1 = self.scene.cpp_world.debug_sphere( *( self.racquet_xyz + 0.2*racquet_face_vector), 0.010, 0x00FF00 )
            self.racquet_face_debug2 = self.scene.cpp_world.debug_sphere( *( self.racquet_xyz + 0.4*racquet_face_vector), 0.015, 0x00FF00 )
            self.racquet_face_debug3 = self.scene.cpp_world.debug_sphere( *( self.racquet_xyz + 0.6*racquet_face_vector), 0.020, 0x00FF00 )

        command_move  = np.array([self.command_body_x - self.body_xyz[0], self.command_body_y - self.body_xyz[1], 0])
        command_move  = np.dot(self.rot_minus_yaw, command_move)
        self.body_dist      = np.linalg.norm(command_move)
        self.body_dist_sqrt = self.body_dist / np.sqrt( 20 + self.body_dist )
        # 0  ->  0/sqrt(20) = 0
        # 1  ->  1/sqrt(21) = 0.22
        # 5  ->  5/sqrt(25) = 1.00
        # 10 -> 10/sqrt(31) = 1.82
        command_move *= self.body_dist_sqrt / self.body_dist

        self.command_hitpos      = self.command_hitpoint - np.array(self.body_xyz)
        self.command_hitpos      = np.dot(self.rot_minus_yaw, self.command_hitpos)
        self.command_hitpos_sqrt = self.command_hitpos / np.sqrt( 20 + np.linalg.norm(self.command_hitpos) )

        self.command_phase          = np.tanh(  (-self.trainer_timeout+15)/10     )   # 10 frames hit duration
        sigma = 4.0                                                                   # -4..4 of those important
        self.command_hit_importance = 1/(sigma*np.sqrt(2*np.pi)) * np.exp( -0.5*((-self.trainer_timeout+15)/sigma)**2 )
        self.command_hit_importance_integral += self.command_hit_importance  # check 1.0 after hit, because dt is one (trainer_timeout changes 1 per frame)
        self.command_phase_pos = self.command_hitpoint + 0.2*self.command_phase*self.command_hitvector  # 0.2 tune arm swing at TYPICAL_BALL_SPEED

        hit2racquet      = np.linalg.norm( self.command_phase_pos - self.racquet_xyz )
        hitspeed2racquet = np.linalg.norm( self.command_hitvector - np.array(self.racquet_v) ) / self.TYPICAL_BALL_SPEED
        self.reward_hit_pos         = self.command_hit_importance * max(0, 1 - hit2racquet)
        self.reward_hit_speed       = self.command_hit_importance * max(0, 1 - hitspeed2racquet)
        self.reward_hit_orientation = self.command_hit_importance * how_good_is_orientation
        self.reward_hit_pos_integral         += self.reward_hit_pos
        self.reward_hit_speed_integral       += self.reward_hit_speed
        self.reward_hit_orientation_integral += self.reward_hit_orientation
        #print(self.command_phase)
        #print(self.command_hit_importance_integral, "pos", self.reward_hit_pos_integral, "speed", self.reward_hit_speed_integral, "good_orientation", self.reward_hit_orientation_integral)

        if self.scene.human_render_detected:
            self.command_phase_debug_sphere = self.scene.cpp_world.debug_sphere( *self.command_phase_pos, self.command_hit_importance + 0.01, 0x805050FF )
        #self.walk_debug = self.scene.cpp_world.debug_sphere( self.walk_target_x, self.walk_target_y, 0.08, 0.08, 0xFFE0E0 )

        if self.training_hit_ball: self.training_hit_ball_counter += 1
        else: self.training_walk_counter += 1

        if reset_potential:
            self.potential = self.calc_potential()  # avoid reward jump on timeout
        return np.concatenate([state, command_move[:2], self.command_hitpos_sqrt, self.command_hitvector_whitened, [self.command_phase]], axis=0)
        #self.command_hitvector_whitened

    def alive_bonus(self, z, pitch):
        #return 100*self.reward_hit_pos + 100*self.reward_hit_speed if z > 0.78 else -100
        #return 100*self.reward_hit_pos + 100*self.reward_hit_speed if z > 0.78 else -200
        #return 2 + 200*self.reward_hit_pos + 200*self.reward_hit_speed if z > 0.78 else -1

        tennis_reward = 800*self.reward_hit_pos + 800*self.reward_hit_speed + 800*self.reward_hit_orientation
        #print("self.reward_hit_pos    %0.2f %0.2f" % (800*self.reward_hit_pos, 800*self.reward_hit_pos_integral))
        #print("self.reward_hit_speed  %0.2f %0.2f" % (800*self.reward_hit_speed, 800*self.reward_hit_speed_integral))

        if z < 0.8:
            if not self.random_lean:
                return -1
            tennis_reward = 0
            self.on_ground_frame_counter += 1
        elif self.on_ground_frame_counter > 0:
            self.on_ground_frame_counter -= 1
        return tennis_reward + self.potential_leak() if self.on_ground_frame_counter<170 else -1

    def potential_leak(self):
        z = self.body_xyz[2]          # 0.00 .. 0.8 .. 1.05 normal walk, 1.2 when jumping
        z = np.clip(z, 0, 0.8)
        return z/0.8 + 1.0            # 1.00 .. 2.0

    def calc_potential(self):
        body_dist = np.linalg.norm( [self.command_body_x - self.body_xyz[0], self.command_body_y - self.body_xyz[1]] )
        #body_to_racquet = [
        #    self.command_hitpoint[0] - self.command_body_x + self.body_xyz[0],
        #    self.command_hitpoint[1] - self.command_body_y + self.body_xyz[1],
        #    self.command_hitpoint[2]
        #    ]
        #if self.scene.human_render_detected:
        #    self.racq_debug = self.scene.cpp_world.debug_sphere( *body_to_racquet, 0.02, 0x80FFFFA0 )
        #racquet_relative_to_body_dist = np.linalg.norm( np.array(body_to_racquet) - self.racquet_xyz )
        #hit2racquet_sqrt = 2*np.sqrt(5) * hit2racquet / np.sqrt( 20 + hit2racquet )  # Taylor series at x==0: x - x^2/40
        face_forward = np.cos(self.angle_to_target)
        #command_progress = 0.3 * (- self.body_dist - racquet_relative_to_body_dist + face_forward) / self.scene.dt
        command_progress = 1.0 * (- self.body_dist + face_forward) / self.scene.dt

        # disable crawl
        if self.body_xyz[2] < 0.8:
            if self.crawl_start_potential is None:
                self.crawl_start_potential = command_progress - self.crawl_ignored_potential
                #print("CRAWL START %+0.1f %+0.1f" % (self.crawl_start_potential, command_progress))
            self.crawl_ignored_potential = command_progress - self.crawl_start_potential
            command_progress  = self.crawl_start_potential
        else:
            #print("CRAWL STOP %+0.1f %+0.1f" % (self.crawl_ignored_potential, command_progress))
            command_progress -= self.crawl_ignored_potential
            self.crawl_start_potential = None

        return command_progress + self.potential_leak()*100
