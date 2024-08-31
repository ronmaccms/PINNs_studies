import os
import warnings

from sympy import Symbol, Eq, Abs
import torch 
import modulus.sym

from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation

from modulus.sym.key import Key

@modulus.sym.main(config_path ="./", config_name= "config.yml")

def run(cfg: ModulusConfig) -> None:
    height = 0.1
    width = 0.1
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((-width/2, -height/2),(width/2, height/2))
    
    ze = ZeroEquation(nu=1e-4, dim=2, time=False, max_distance= height/2)
    ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=2, time=False)
    
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y")],
        output_keys = [Key("u"), Key("v"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )
    
    nodes= ( ns.make_nodes() + ze.make_nodes() + [flow_net.make_node(name="flow_network")])
    
    cavity_domain = Domain()
    
    top_wall = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= rec,
        outvar = {"u": 1.5, "v": 0},
        batch_size = cfg.batch_size.TopWall,
        lambda_weighting = {"u":1.0-20*Abs(x), "v":1.0},
        criteria = Eq(y, height/2),
    )
    cavity_domain.add_constraint(top_wall,"top_wall")
    
    no_slip = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec,
        outvar = {"u": 0, "v": 0},
        batch_size = cfg.batch_size.NoSlip,
        criteria= y < height /2,
    )
    cavity_domain.add_constraint(no_slip,"no_slip")
    
    
    interior = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = rec,
        outvar = {"continuity":0,"momentum_x":0,"momentum_y":0},
        batch_size = cfg.batch_size.Interior,
        compute_sdf_derivatives= True,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            },
    )
    cavity_domain.add_constraint(interior,"interior")
    
    slv = Solver(cfg, cavity_domain)
    
    slv.solve()
    
    
if __name__ == "__main__":
    run()

