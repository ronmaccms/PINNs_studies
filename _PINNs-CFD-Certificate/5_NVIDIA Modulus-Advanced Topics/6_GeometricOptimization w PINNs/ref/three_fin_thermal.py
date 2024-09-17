
import os
import warnings

import torch
from sympy import Symbol, Eq, Abs, tanh, Or, And
import itertools
import numpy as np

import modulus.sym
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.hydra import to_absolute_path, instantiate_arch
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box, Channel, Plane
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node

from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec, GradNormal
from modulus.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion

from three_fin_geometry import *

@modulus.sym.main(config_path="./", config_name="conf_thermal.yaml")

def run(cfg: ModulusConfig) -> None:
    ad = AdvectionDiffusion(T = "theta_f", rho=1.0, D=0.02, dim = 3, time=False)
    dif = Diffusion(T = "theta_s", D = 0.0625, dim = 3, time= False)
    dif_interface = DiffusionInterface("theta_f","theta_s", 1.0,5.0, dim=3, time= False)
    f_grad = GradNormal("theta_f", dim=3, time=False)
    s_grad = GradNormal("theta_s", dim=3, time=False)
    
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
    
    thermal_f_net = FullyConnectedArch(
        input_keys= input_keys, output_keys= [Key("theta_f")]
    )
    
    thermal_s_net = FullyConnectedArch(
        input_keys= input_keys, output_keys= [Key("theta_s")]
    )
    
    thermal_nodes = (
        ad.make_nodes()
        + dif.make_nodes()
        + dif_interface.make_nodes()
        + f_grad.make_nodes()
        + s_grad.make_nodes()
        + [flow_net.make_node(name="flow_net", optimize = False)]
        + [thermal_f_net.make_node(name= "thermal_f_network")]
        + [thermal_s_net.make_node(name= "thermal_s_network")]
    )
    
    geo = ThreeFin()
    
    inlet_t = 293.15/ 273.15 - 1
    grad_t = 360/ 273.15
    
    thermal_domain = Domain()
    
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes = thermal_nodes,
        geometry = geo.inlet,
        outvar = {"theta_f": inlet_t},
        batch_size = cfg.batch_size.Inlet,
        criteria = Eq(x, channel_origin[0]),
        lambda_weighting = {"theta_f" : 1.0},
        parameterization = geo.pr,
    )
    thermal_domain.add_constraint(constraint_inlet, "inlet")
    
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes =  thermal_nodes,
        geometry = geo.outlet,
        outvar = {"normal_gradient_theta_f": 0},
        batch_size = cfg.batch_size.Outlet,
        criteria = Eq(x, channel_origin[0] + channel_dim[0]),
        lambda_weighting = {"normal_gradient_theta_f" : 1.0},
        parameterization = geo.pr,
    )
    thermal_domain.add_constraint(constraint_outlet, "outlet")
    
    def wall_criteria(invar, params):
        sdf = geo.three_fin.sdf(invar, params)
        return np.less(sdf["sdf"], -1e-5)
    
    channel_walls = PointwiseBoundaryConstraint(
        nodes = thermal_nodes,
        geometry = geo.channel,
        outvar = {"normal_gradient_theta_f": 0},
        batch_size = cfg.batch_size.ChannelWalls,
        criteria = wall_criteria,
        lambda_weighting = {"normal_gradient_theta_f" : 1.0},
        parameterization = geo.pr,
    )
    thermal_domain.add_constraint(channel_walls, "channel_walls")
    
    def interface_criteria(invar, params):
        sdf = geo.three_fin.sdf(invar, params)
        return np.greater(sdf["sdf"],0)
        
    fluid_solid_interface = PointwiseBoundaryConstraint(
        nodes = thermal_nodes,
        geometry = geo.three_fin,
        outvar = {
            "diffusion_interface_dirichlet_theta_f_theta_s" : 0,
            "diffusion_interface_neumann_theta_f_theta_s" : 0,
        },
        batch_size = cfg.batch_size.SolidInterface,
        criteria = interface_criteria,
        parameterization = geo.pr,
    )
    thermal_domain.add_constraint(fluid_solid_interface, "fluid_solid_interface")
    
    sharpen_tanh = 60.0
    source_func_xl = (tanh(sharpen_tanh * (x - source_origin[0])) + 1.0) / 2.0
    source_func_xh = (
        tanh(sharpen_tanh * ((source_origin[0] + source_dim[0]) - x)) + 1.0
    ) / 2.0
    source_func_zl = (tanh(sharpen_tanh * (z - source_origin[2])) + 1.0) / 2.0
    source_func_zh = (
        tanh(sharpen_tanh * ((source_origin[2] + source_dim[2]) - z)) + 1.0
    ) / 2.0
    gradient_normal = (
        grad_t * source_func_xl * source_func_xh * source_func_zl * source_func_zh
    )
    
    heat_source = PointwiseBoundaryConstraint(
        nodes = thermal_nodes,
        geometry = geo.three_fin,
        outvar = {"normal_gradient_theta_s" : gradient_normal},
        batch_size = cfg.batch_size.HeatSource,
        criteria = Eq (y, source_origin[1]),
    )
    thermal_domain.add_constraint(heat_source, "heat_source")
    
    lr_flow_interior = PointwiseInteriorConstraint(
        nodes= thermal_nodes,
        geometry = geo.geo,
        outvar = {"advection_diffusion_theta_f":0},
        batch_size = cfg.batch_size.InteriorLR,
        criteria = Or (x < -1.1, x > 0.5),
    )
    thermal_domain.add_constraint(lr_flow_interior, "lr_flow_interior")
    
    hr_flow_interior = PointwiseInteriorConstraint(
        nodes= thermal_nodes,
        geometry = geo.geo,
        outvar = {"advection_diffusion_theta_f":0},
        batch_size = cfg.batch_size.InteriorHR,
        criteria = And (x > -1.1, x < 0.5),
    )
    thermal_domain.add_constraint(hr_flow_interior, "hr_flow_interior")
    
    
    solid_interior = PointwiseInteriorConstraint(
        nodes= thermal_nodes,
        geometry = geo.three_fin,
        outvar = {"diffusion_theta_s":0},
        batch_size = cfg.batch_size.SolidInterior,
        lambda_weighting = {"diffusion_theta_s": 100.0},
    )
    thermal_domain.add_constraint(solid_interior, "solid_interior")
    
    
    
    thermal_slv = Solver(cfg, thermal_domain)
    
    thermal_slv.solve()
    
    
    
if __name__ == "__main__":
    run()
