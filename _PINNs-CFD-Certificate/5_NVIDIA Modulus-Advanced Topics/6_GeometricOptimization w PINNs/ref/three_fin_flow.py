
import os 
import warnings

import torch
from torch.utils.data import DataLoader, Dataset
from sympy import Symbol, Eq, Abs, tanh, Or, And
import numpy as np
import itertools

import modulus.sym 
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box, Channel, Plane
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node

from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.basic import NormalDotVec, GradNormal
from modulus.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from modulus.sym.models.fully_connected import FullyConnectedArch

from three_fin_geometry import *

@modulus.sym.main(config_path = "./", config_name="conf_flow.yaml")

def run(cfg: ModulusConfig) -> None:
    if cfg.custom.turbulent:
        ze = ZeroEquation(nu = 0.002, dim=3, time=False, max_distance=0.5)
        ns = NavierStokes(nu = ze.equations["nu"], rho=1.0, dim=3, time= False)
        navier_stokes_nodes= ns.make_nodes() + ze.make_nodes()
    else:
        ns = NavierStokes(nu = 0.01, rho=1.0, dim=3, time= False)
        navier_stokes_nodes= ns.make_nodes()
        
    #input_keys = [Key("x"), Key("y"), Key("z")]
    
    ##########################################
    input_keys = [
        Key("x"), 
        Key("y"), 
        Key("z"),
        Key("fin_height_m"),
        Key("fin_height_s"),
        Key("fin_length_m"),
        Key("fin_length_s"),
        Key("fin_thickness_m"),
        Key("fin_thickness_s"),
    ]
    ##########################################
    
    flow_net = FullyConnectedArch(
        input_keys = input_keys, output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
    )
    
    normal_dot_vel = NormalDotVec()
    
    flow_nodes = (
        navier_stokes_nodes
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name= "flow_network")]
    )
    
    geo = ThreeFin()
    
    inlet_vel = 1.0
    volumetric_flow = 1.0
    
    flow_domain = Domain()
    
    u_profule = inlet_vel * tanh((0.5 - Abs(y))/ 0.02) * tanh((0.5 - Abs(z))/ 0.02)
    
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes = flow_nodes,
        geometry = geo.inlet,
        outvar = {"u": u_profule, "v": 0, "w":0},
        batch_size = cfg.batch_size.Inlet,
        criteria = Eq(x, channel_origin[0]),
        lambda_weighting = {
            "u": 1.0,
            "v": 1.0,
            "w": 1.0,
        },
        parameterization = geo.pr,
        batch_per_epoch = 5000,
    )
    flow_domain.add_constraint(constraint_inlet, "inlet")
    
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes = flow_nodes,
        geometry = geo.outlet,
        outvar = {"p": 0},
        batch_size = cfg.batch_size.Outlet,
        criteria = Eq(x, channel_origin[0]+ channel_dim[0]),
        lambda_weighting = {"p": 1.0},
        parameterization = geo.pr,
        batch_per_epoch = 5000,
    )
    flow_domain.add_constraint(constraint_outlet, "outlet")


    
    no_slip = PointwiseBoundaryConstraint(
        nodes = flow_nodes,
        geometry = geo.geo,
        outvar = {"u": 0, "v": 0, "w":0},
        batch_size = cfg.batch_size.NoSlip,
        lambda_weighting = {
            "u": 1.0,
            "v": 1.0,
            "w": 1.0,
        },
        parameterization = geo.pr,
        batch_per_epoch = 5000,
    )
    flow_domain.add_constraint(no_slip, "no_slip")
    
    lr_interior = PointwiseInteriorConstraint(
        nodes = flow_nodes,
        geometry = geo.geo,
        outvar = {"continuity": 0, "momentum_x": 0, "momentum_y":0,"momentum_z":0},
        batch_size = cfg.batch_size.InteriorLR,
        lambda_weighting = {
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives= True,
        parameterization = geo.pr,
        batch_per_epoch = 5000,
        criteria = Or(x < -1.1, x > 0.5),
    )
    flow_domain.add_constraint(lr_interior, "lr_interior")
    
    hr_interior = PointwiseInteriorConstraint(
        nodes = flow_nodes,
        geometry = geo.geo,
        outvar = {"continuity": 0, "momentum_x": 0, "momentum_y":0,"momentum_z":0},
        batch_size = cfg.batch_size.InteriorHR,
        lambda_weighting = {
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives= True,
        parameterization = geo.pr,
        batch_per_epoch = 5000,
        criteria = And (x > -1.1, x < 0.5),
    )
    flow_domain.add_constraint(hr_interior, "hr_interior")
    
    def integral_criteria(invar, params):
        sdf = geo.geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)
        
    integral_continuity = IntegralBoundaryConstraint(
        nodes = flow_nodes,
        geometry = geo.integral_plane,
        outvar = {"normal_dot_vel":volumetric_flow},
        batch_size = 5,
        integral_batch_size = cfg.batch_size.IntegralContinuity,
        criteria = integral_criteria,
        lambda_weighting = {"normal_dot_vel": 1.0},
        parameterization = {**geo.pr, **{x_pos: (-1.1,0.1)}},
        fixed_dataset = False,
        num_workers = 4,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")
    
    
    invar_inlet_pressure = geo.integral_plane.sample_boundary(
        1024, parameterization= {**fixed_param_ranges, **{x_pos: -2}}
    )


    pressure_monitor = PointwiseMonitor(
        invar_inlet_pressure,
        output_names = ["p"],
        metrics = {"inlet_pressure": lambda var: torch.mean(var["p"])},
        nodes = flow_nodes,
    )
    flow_domain.add_monitor(pressure_monitor)
    
    flow_slv = Solver(cfg, flow_domain)

    flow_slv.solve()
    
if __name__ == "__main__":
    run()