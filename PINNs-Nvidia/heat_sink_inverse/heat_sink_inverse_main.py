import os
import sys
import warnings

import torch
import numpy as np
from sympy import Symbol, Eq

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.utils.io import csv_to_dict

from modulus.sym.eq.pdes.navier_stokes import NavierStokes, GradNormal
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint, IntegralBoundaryConstraint, PointwiseConstraint

from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.validator import PointwiseValidator

from modulus.sym.key import Key
from modulus.sym.node import Node

@modulus.sym.main(config_path="./", config_name="config.yml")

def run(cfg: ModulusConfig) -> None:
    nu, D = Symbol("nu"), Symbol("D")

    ns = NavierStokes(nu=nu, rho=1, dim=2, time=False)
    ade = AdvectionDiffusion(T="c", rho=1, D=D, dim=2, time=False)

    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y")],
        output_keys = [Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    heat_net = instantiate_arch(
        input_keys = [Key("x"), Key("y")],
        output_keys = [Key("c")],
        cfg=cfg.arch.fully_connected,
    )

    invert_net_nu = instantiate_arch(
        input_keys = [Key("x"), Key("y")],
        output_keys = [Key("nu")],
        cfg=cfg.arch.fully_connected,
    )

    invert_net_D = instantiate_arch(
        input_keys = [Key("x"), Key("y")],
        output_keys = [Key("D")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        ns.make_nodes(
            detach_names=[
                "u",
                "u__x",
                "u__x__x",
                "u__y",
                "u__y__y",
                "v",
                "v__x",
                "v__x__x",
                "v__y",
                "v__y__y",
                "p",
                "p__x",
                "p__y",
            ]
        )
        + ade.make_nodes(
            detach_names=["u", "v", "c", "c__x", "c__y", "c__x__x", "c__y__y"]
        )
        + [flow_net.make_node(name="flow_network")]
        + [heat_net.make_node(name="heat_network")]
        + [invert_net_nu.make_node(name="invert_net_nu")]
        + [invert_net_D.make_node(name="invert_net_D")]
    )

    base_temp = 293.498 # given value from sample project or file

    mapping = {
        "Points:0" : "x",
        "Points:1" : "y",
        "U:0" : "u",
        "U:1" : "v",
        "p" : "p",
        "T" : "c",
    }
    openfoam_var = csv_to_dict(
        to_absolute_path("./heat_sink_Pr5_clipped2.csv"), mapping
    )
    openfoam_var["c"]= openfoam_var["c"]/base_temp-1.0

    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u", "v", "p", "c"]
    }
    openfoam_outvar_numpy["continuity"] = np.zeros_like(openfoam_outvar_numpy["u"])
    openfoam_outvar_numpy["momentum_x"] = np.zeros_like(openfoam_outvar_numpy["u"])
    openfoam_outvar_numpy["momentum_y"] = np.zeros_like(openfoam_outvar_numpy["u"])
    openfoam_outvar_numpy["advection_diffusion_c"] = np.zeros_like(openfoam_outvar_numpy["u"])

    domain = Domain()

    data = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        outvar=openfoam_outvar_numpy,
        batch_size=cfg.batch_size.data,
    )
    domain.add_constraint(data, "interior_data")

    monitor = PointwiseMonitor(
        openfoam_invar_numpy,
        output_names=["nu"], # nu is the viscosity values
        metrics={"mean_nu": lambda var: torch.mean(var["nu"])},
        nodes=nodes,
    )
    domain.add_monitor(monitor)

    monitor = PointwiseMonitor(
        openfoam_invar_numpy,
        output_names=["D"], 
        metrics={"mean_D": lambda var: torch.mean(var["D"])},
        nodes=nodes,
    )
    domain.add_monitor(monitor)

    slv = Solver(cfg, domain)

    slv.solve()

if __name__ == "__main__":
    run()