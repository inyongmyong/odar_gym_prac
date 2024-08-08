from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from base_task import BaseTask
from aerial_robot_config import AerialRobotCfg
from controllers.controller import Controller

import matplotlib.pyplot as plt
from aerial_gym.utils.helpers import asset_class_to_AssetOptions
import time

import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

class AerialRobot(BaseTask):

    def __init__(self, cfg: AerialRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.debug_viz = False
        num_actors = 1

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        bodies_per_env = self.robot_num_bodies

        self.vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, num_actors, 13)

        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]
                
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()

        self.counter = 0

        self.action_upper_limits = torch.tensor(
            [1, 1, 1, 1], device=self.device, dtype = torch.float32)
        self.action_lower_limits = torch.tensor(
            [-1, -1, -1, -1], device=self.device, dtype=torch.float32)
        
        # control tensors
        self.action_input = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)

        self.controller = Controller(self.cfg.control, self.device)

        self.desired_pos = [0,0,2]
        quat = euler_to_quaternion(0,0,0)
        self.desired_rot = quat
        self.des_buf[:,0:3] = torch.tensor(self.desired_pos, device=self.device, dtype=torch.float)
        self.des_buf[:,3:7] = torch.tensor(self.desired_rot, device=self.device, dtype=torch.float)

        self.grav_upward = -1*torch.tensor(self.cfg.sim.gravity, device=self.device)

        self.k_p = torch.diag(
            torch.tensor(self.cfg.control.k_p, dtype=torch.float32, device=self.device)
        )
        self.k_R = torch.diag(
            torch.tensor(self.cfg.control.k_R, dtype=torch.float32, device=self.device)
        )
        self.k_v = torch.diag(
            torch.tensor(self.cfg.control.k_v, dtype=torch.float32, device=self.device)
        )
        self.k_w = torch.diag(
            torch.tensor(self.cfg.control.k_w, dtype=torch.float32, device=self.device)
        )
        self.k_i = torch.diag(
            torch.tensor(self.cfg.control.k_i, dtype=torch.float32, device=self.device)
        )
        self.k_anti_windup = torch.diag(
            torch.tensor(
                self.cfg.control.k_anti_windup, dtype=torch.float32, device=self.device
            )
        )



        # if self.viewer:
        #     cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
        #     cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
        #     cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
        #     cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
        #     cam_ref_env = self.cfg.viewer.ref_env
            
        #     self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.cfg.env.num_envs, device=self.sim_device, dtype=torch.long)

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def _create_envs(self):
        asset_root = "robot"
        asset_file = "robot.urdf"

        asset_options = gymapi.AssetOptions()
        
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        
        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))) 
            pos = torch.tensor([0,0,1], device = self.device)
            start_pose.p = gymapi.Vec3(*pos)
            start_pose.r = gymapi.Quat.from_euler_zyx(0,0,0)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.robot_asset.name, i, self.cfg.robot_asset.collision_mask, 0)
            
            self.robot_body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # prop_names = self.cfg.control.prop_names
        # prop1 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[0], gymapi.DOMAIN_SIM
        #     )
        # prop2 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[1], gymapi.DOMAIN_SIM
        #     )
        # prop3 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[2], gymapi.DOMAIN_SIM
        #     )
        # prop4 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[3], gymapi.DOMAIN_SIM
        #     )
        # prop5 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[4], gymapi.DOMAIN_SIM
        #     )
        # prop6 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[5], gymapi.DOMAIN_SIM
        #     )
        # prop7 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[6], gymapi.DOMAIN_SIM
        #     )
        # prop8 = self.gym.find_actor_rigid_body_index(
        #         self.envs[0], self.actor_handles[0], prop_names[7], gymapi.DOMAIN_SIM
        #     )
        
        # self.prop_idx = [prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8]


        self.robot_mass = 0
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
        self.inertia = []
        self.inertia.append(
                Mat33_to_numpy(self.robot_body_props[0].inertia)
            )  # 0 means mother link
        self.inertia = np.array(self.inertia).reshape(self.num_envs, 3, 3)
        self.inertia = torch.tensor(
            self.inertia, dtype=torch.float32, device=self.device
        )

    def step2(self, actions):
        self.forces[:,0,0]=20
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.forces),
            gymtorch.unwrap_tensor(self.torques),
            gymapi.LOCAL_SPACE,
        )
        self.post_physics_step()
        print(self.root_states)

    def step(self, actions):
        self.render()
        
        forces, self.torques[:,0,0:3], self.rot_e_integral_buf, R_current = self.compute_forces()
        thrust = self.compute_thrust(forces, self.torques[:,0,0:3], R_current)
        
        self.forces[:,1:9,2] = thrust

        # j=0
        # for i in self.prop_idx:
        #     self.forces[:,i,2] = thrust[:,j]
        #     j+=1

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.forces),
            None,
            gymapi.LOCAL_SPACE,
        )
        # self.gym.apply_rigid_body_force_tensors(
        #     self.sim,
        #     None,
        #     gymtorch.unwrap_tensor(self.torques),
        #     gymapi.GLOBAL_SPACE, 
        # )
        self.gym.simulate(self.sim)
        self.post_physics_step()
        
        # print(thrust)

    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim) #RL 용 post_physics


    def compute_forces(self):
        #변수 생성
        forces = self.forces
        torques = self.torques
        rot_e_integral = self.rot_e_integral_buf
        current_state = self.root_states
        desired_state = self.des_buf

        #current state 할당
        p_current = current_state[:,0:3]
        R_current = current_state[:,3:7]
        v_current = current_state[:,7:10]
        w_current = current_state[:,10:13]

        #desired state 할당
        p_desired = desired_state[:,0:3]
        R_desired = desired_state[:,3:7]
        v_desired = desired_state[:,7:10]
        w_desired = desired_state[:,10:13]
        a_desired = desired_state[:,13:16]
        alpha_desired = desired_state[:,16:19]

        # #traj_control 업데이트
        # p_desired = traj_control(p_current, p_desired)
        # R_desired = traj_control(R_current, R_desired)
        # v_desired = traj_control(v_current, v_desired)
        # w_desired = traj_control(w_current, w_desired)

        #impedance control
        e_p = p_desired - p_current
        R_current = quaternion_to_SO3(R_current)
        R_desired = quaternion_to_SO3(R_desired)
        e_R = unskew_torch(
            torch.transpose(R_desired, 1, 2) @ R_current
            - torch.transpose(R_current, 1, 2) @ R_desired,
            self.device,
        )  # (E,3)
        
        # print(e_R[0,:])
        # print(R_desired[0,:])
        # print(R_current[0,:])

        e_v = v_desired - v_current

        w_current = torch.squeeze(torch.transpose(R_current,1,2) @ w_current.view(-1,3,1), -1)
        w_desired = torch.squeeze(torch.transpose(R_current,1,2) @ w_desired.view(-1,3,1), -1)
        e_w = w_current - torch.squeeze(
            torch.transpose(R_current, 1, 2) @ R_desired @ w_desired.view(-1, 3, 1), -1
        )

        force_desired = torch.squeeze(
            torch.transpose(R_current, 1, 2)
            @ (
                self.robot_mass * (self.grav_upward.view(-1, 3, 1) + a_desired.view(-1,3,1))
                + self.k_v @ e_v.view(-1, 3, 1)
                + self.k_p @ e_p.view(-1, 3, 1)
            ),
            -1,
        )
        # force_desired[:,0] = -force_desired[:,0]
        # force_desired[:,1] = -force_desired[:,1]
        # force_desired[:,2] = -force_desired[:,2]
        force_desired = torch.squeeze(R_current @ force_desired.view(-1,3,1),-1)

        skew_w_current = skew_torch(w_current, self.device)  # (E,3,3)
        torque_desired = torch.squeeze(
            skew_w_current @ self.inertia @ w_current.view(-1, 3, 1)
            - self.k_w @ e_w.view(-1, 3, 1)
            - self.k_R @ e_R.view(-1, 3, 1)
            - self.k_i @ rot_e_integral.view(-1, 3, 1)
            - self.inertia @ (
                skew_w_current @ R_desired @ w_desired.view(-1, 3, 1)
                - R_desired @ alpha_desired.view(-1, 3, 1)
            ),
            -1,
        )  # (E,3)
        torque_desired = torch.squeeze(R_current @ torque_desired.view(-1,3,1),-1)
        # torque_desired[:, 2] = -1*torque_desired[:, 2]
        # force_desired[:, 0] = -1*force_desired[:, 0]
        # torque_desired[:, 1] = -1*torque_desired[:, 1]

        # forces = forces.view(self.num_envs, self.robot_num_bodies, 3)
        # torques = torques.view(self.num_envs, self.robot_num_bodies, 3)

        rot_e_integral += (
            e_w + self.cfg.control.gamma * e_R
        ) * self.dt

        return force_desired, torque_desired, rot_e_integral, R_current
    
    def compute_thrust(self, force, torque, R_current):
        #U value
        u1 = [0.68,0.28,0.68] 
        u2 = [0.68,0.28,-0.68] 
        u3 = [0.68,-0.28,0.68] 
        u4 = [0.68,-0.28,-0.68]

        u1 = torch.tensor(u1, device=self.device, dtype=float).view(3,-1)
        u2 = torch.tensor(u2, device=self.device, dtype=float).view(3,-1)
        u3 = torch.tensor(u3, device=self.device, dtype=float).view(3,-1)
        u4 = torch.tensor(u4, device=self.device, dtype=float).view(3,-1)
        u5 = u1
        u6 = u2
        u7 = u3
        u8 = u4

        #R value
        t1 = [-0.15, -0.15, 0.24]
        t2 = [-0.18, 0.38, 0.01]
        t3 = [-0.18, -0.38, -0.01]
        t4 = [-0.18, 0.16, -0.21]

        t1 = torch.tensor(t1, device=self.device, dtype=float).view(3,-1)
        t2 = torch.tensor(t2, device=self.device, dtype=float).view(3,-1)
        t3 = torch.tensor(t3, device=self.device, dtype=float).view(3,-1)
        t4 = torch.tensor(t4, device=self.device, dtype=float).view(3,-1)
        t5 = -t1
        t6 = -t2
        t7 = -t3
        t8 = -t4

        #B matrix 생성1
        B_f = torch.cat((u1,u2,u3,u4,u5,u6,u7,u8), dim=1)
        B_t = torch.cat((t1,t2,t3,t4,t5,t6,t7,t8), dim=1)
        B_ft = torch.cat((B_f,B_t), dim=0)

        #B pseudo inverse 생성
        B_transpose = torch.transpose(B_ft, 0, 1)
        B_pseudo = B_transpose @ torch.inverse(B_ft @ B_transpose)

        B_matrix = torch.zeros((self.num_envs,8,6), device=self.device, dtype=float)
        B_matrix[:,0:8,0:6] = B_pseudo
        
        #U matrix 생성
        force = torch.squeeze(torch.transpose(R_current,1,2) @ force.view(-1,3,1), -1)
        torque = torch.squeeze(torch.transpose(R_current,1,2) @ torque.view(-1,3,1), -1)
        U_matrix = torch.zeros((self.num_envs,6,1), device=self.device, dtype=float)
        U_matrix[:,0:3,0] = force
        U_matrix[:,3:6,0] = torque

        #thrust 계산
        thrust = torch.squeeze(B_pseudo @ U_matrix, -1)

        return thrust


#traj_control 코드
def traj_control(current, desired):
    traj_desired = current + (desired - current)/5
    
    return traj_desired

#from odar_gym utils + chat GPT 
def unskew_torch(SO3, dev):
    w1 = SO3[:,2,1]
    w2 = SO3[:,0,2]
    w3 = SO3[:,1,0]
    v = torch.zeros((SO3.size(0),3), device=dev)
    v[:,0] = w1
    v[:,1] = w2
    v[:,2] = w3
    return v

def skew_torch(v, dev):
    SO3 = torch.zeros((v.size(0), 3, 3), dtype=torch.float32, device=dev)
    SO3[:, 0, 1] = -v[:, 2]
    SO3[:, 0, 2] = v[:, 1]
    SO3[:, 1, 0] = v[:, 2]
    SO3[:, 1, 2] = -v[:, 0]
    SO3[:, 2, 0] = -v[:, 1]
    SO3[:, 2, 1] = v[:, 0]
    return SO3

def quaternion_to_SO3(quaternions):
    """
    Convert quaternions to SO(3) rotation matrices.
    
    Args:
        quaternions (torch.Tensor): Tensor of shape (N, 4) representing N quaternions.
        
    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) representing N rotation matrices.
    """
    # Normalize the quaternions
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    
    # Extract the individual components
    w = quaternions[:, 3]
    x = quaternions[:, 0]
    y = quaternions[:, 1]
    z = quaternions[:, 2]
    
    # Compute the rotation matrix elements
    R = torch.zeros((quaternions.size(0), 3, 3), device=quaternions.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    
    Parameters:
        roll: Rotation around the x-axis in radians.
        pitch: Rotation around the y-axis in radians.
        yaw: Rotation around the z-axis in radians.
    
    Returns:
        A list containing the quaternion [w, x, y, z].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]

##############################################################################################

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passe1d on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def Mat33_to_numpy(Mat):

    row_x = Mat.x
    row_y = Mat.y
    row_z = Mat.z

    x = Vec3_to_numpy(row_x)
    y = Vec3_to_numpy(row_y)
    z = Vec3_to_numpy(row_z)

    return np.array([x, y, z])

def Vec3_to_numpy(Vec):
    x = Vec.x
    y = Vec.y
    z = Vec.z

    return np.array([x, y, z]).transpose()

def main():
    args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])
    
    env_cfg = AerialRobotCfg
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    env = AerialRobot(cfg=env_cfg,
                      sim_params=sim_params,
                      physics_engine=args.physics_engine,
                      sim_device=args.sim_device,
                      headless=False)          
    
    actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))

    while not env.gym.query_viewer_has_closed(env.viewer):
        env.step(actions)
        # env.render()
        # env.gym.simulate(env.sim)

if __name__ == "__main__":
    main()
        