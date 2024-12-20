from sympy import Symbol, Eq

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain 

from modulus.sym.geometry.primitives_2d import Rectangle, Circle

from modulus.sym.domain.constraint import(
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer

from modulus.sym.key import Key
from modulus.sym.node import Node

from modulus.sym.eq.pdes.linear_elasticity import LinearElasticityPlaneStress

@modulus.sym.main(config_path="./", config_name="config.yml")

def run(cfg: ModulusConfig) -> None:

    E = 73.0 * 10**9
    nu = 0.33

    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu_real = E / (2 * (1 + nu))

    lambda_ = lambda_ / mu_real
    mu = 1.0

    le = LinearElasticityPlaneStress(lambda_= lambda_, mu= mu)

    elasticity_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("sigma_hoop")],
        output_keys = [
            Key("u"),
            Key("v"),
            Key("sigma_xx"),
            Key("sigma_yy"),
            Key("sigma_xy"),
        ],
        cfg = cfg.arch.fully_connected
    )

    nodes = le.make_nodes() + [elasticity_net.make_node(name="elasticity_network")]

    # DEFINE THE GEOMETRY OF THE WINDOW AND PANEL 
    x, y, sigma_hoop = Symbol("x"), Symbol("y"), Symbol("sigma_hoop")

    panel_origin = (-0.5, -0.9)
    panel_dim = (1, 1.8)
    window_origin = (-0.125, -0.2)
    window_dim = (0.25, 0.4)

    # creating the fillet for the overlap of triangles to get the round corners for the window

    # rectangle 1
    panel_aux1_origin = (-0.075, -0.2)
    panel_aux1_dim = (0.15, 0.4)
    # rectangle 2
    panel_aux2_origin = (-0.125, -0.15)
    panel_aux2_dim = (0.25, 0.3)

    circle_nw_center = (-0.075, 0.15)
    circle_ne_center = (0.075, 0.15)
    circle_se_center = (0.075, -0.15)
    circle_sw_center = (-0.075, -0.15)
    circle_radius = 0.05

    # CREATE HIGH-RES AND LOW-RES AREAS IN SPECIFIC AREAS OF THE SIMULATION
    hr_zone_origin = (-0.2, -0.4)
    hr_zone_dim = (0.4, 0.8)

    # CREATE THE GEOMETRY
    panel = Rectangle(
        panel_origin, (panel_origin[0] + panel_dim[0], panel_origin[1] + panel_dim[1])
    )
    window = Rectangle(
        window_origin, (window_origin[0] + window_dim[0], window_origin[1] + window_dim[1])
    )
    panel_aux1 = Rectangle(
        panel_aux1_origin, (panel_aux1_origin[0] + panel_aux1_dim[0], panel_aux1_origin[1] + panel_aux1_dim[1])
    )
    panel_aux2 = Rectangle(
        panel_aux2_origin, (panel_aux2_origin[0] + panel_aux2_dim[0], panel_aux2_origin[1] + panel_aux2_dim[1])
    )
    hr_zone = Rectangle(
        hr_zone_origin, (hr_zone_origin[0] + hr_zone_dim[0], hr_zone_origin[1] + hr_zone_dim[1])
    )
    circle_nw = Circle(circle_nw_center, circle_radius)
    circle_ne = Circle(circle_ne_center, circle_radius)
    circle_se = Circle(circle_se_center, circle_radius)
    circle_sw = Circle(circle_sw_center, circle_radius)

    corners = (
        window - panel_aux1 - panel_aux2 - circle_nw - circle_ne - circle_se - circle_sw
    )
    window = window - corners
    geo = panel - window

    hr_geo = geo & hr_zone

    characteristic_length = panel_dim[0]
    characteristic_disp = 0.001 * window_dim[0]

    sigma_normalization = characteristic_length / (mu_real * characteristic_disp)
    sigma_hoop_lower = 46 * 10**6 * sigma_normalization
    sigma_hoop_upper = 56.5 * 10**6 * sigma_normalization
    sigma_hoop_range = (sigma_hoop_lower, sigma_hoop_upper)
    param_ranges = {sigma_hoop: sigma_hoop_range}

    inference_param_ranges = {sigma_hoop: 46 * 10**6 * sigma_normalization}

    bounds_x = (panel_origin[0], panel_origin[0] + panel_dim[0])
    bounds_y = (panel_origin[1], panel_origin[1] + panel_dim[1])
    hr_bounds_x = (hr_zone_origin[0], hr_zone_origin[0] + hr_zone_dim[0])
    hr_bounds_y = (hr_zone_origin[1], hr_zone_origin[1] + hr_zone_dim[1])

    # DEFINING THE B.C INTERIOR CONSTRAINTS
    domain = Domain()

    panel_left = PointwiseBoundaryConstraint(
        nodes= nodes,
        geometry= geo,
        outvar= {"traction_x": 0.0, "traction_y": 0.0},
        batch_size= cfg.batch_size.panel_left,
        criteria= Eq(x, panel_origin[0]),
        parameterization= param_ranges,        
    )
    domain.add_constraint(panel_left, "panel_left")

    panel_right = PointwiseBoundaryConstraint(
        nodes= nodes,
        geometry= geo,
        outvar= {"traction_x": 0.0, "traction_y": 0.0},
        batch_size= cfg.batch_size.panel_right,
        criteria= Eq(x, panel_origin[0] + panel_dim[0]),
        parameterization= param_ranges,        
    )
    domain.add_constraint(panel_right, "panel_right")

    panel_bottom = PointwiseBoundaryConstraint(
        nodes= nodes,
        geometry= geo,
        outvar= {"v": 0.0},
        batch_size= cfg.batch_size.panel_bottom,
        criteria= Eq(y, panel_origin[1]),
        parameterization= param_ranges,        
    )
    domain.add_constraint(panel_bottom, "panel_bottom")

    panel_corner = PointwiseBoundaryConstraint(
        nodes= nodes,
        geometry= geo,
        outvar= {"u": 0.0},
        batch_size= cfg.batch_size.panel_corner,
        criteria= Eq(x, panel_origin[0]) & (y > panel_origin[1]) & (y < panel_origin[1] + 1e-3),
        parameterization= param_ranges,        
    )
    domain.add_constraint(panel_corner, "panel_corner")

    panel_top = PointwiseBoundaryConstraint(
        nodes= nodes,
        geometry= geo,
        outvar= {"traction_x": 0.0, "traction_y": sigma_hoop},
        batch_size= cfg.batch_size.panel_top,
        criteria= Eq(x, panel_origin[0]),
        parameterization= param_ranges,        
    )
    domain.add_constraint(panel_top, "panel_top")

    panel_window = PointwiseBoundaryConstraint(
        nodes= nodes,
        geometry= window,
        outvar= {"traction_x": 0.0, "traction_y": 0.0},
        batch_size= cfg.batch_size.panel_window,
        parameterization= param_ranges,        
    )
    domain.add_constraint(panel_window, "panel_window")

    lr_interior = PointwiseInteriorConstraint(
        nodes= nodes,
        geometry= geo,
        outvar= {
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_xy": 0.0,
        },
        batch_size= cfg.batch_size.lr_interior,
        bounds= {x: bounds_x, y: bounds_y},
        lambda_weighting = {
            "equilibrium_x": Symbol("sdf"),
            "equilibrium_y": Symbol("sdf"),
            "stress_disp_xx": Symbol("sdf"),
            "stress_disp_yy": Symbol("sdf"),
            "stress_disp_xy": Symbol("sdf"),
        },
        parameterization= param_ranges,
    )
    domain.add_constraint(lr_interior, "lr_interior")

    hr_interior = PointwiseInteriorConstraint(
        nodes= nodes,
        geometry= geo,
        outvar= {
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_xy": 0.0,
        },
        batch_size= cfg.batch_size.hr_interior,
        bounds= {x: hr_bounds_x, y: hr_bounds_y},
        lambda_weighting = {
            "equilibrium_x": Symbol("sdf"),
            "equilibrium_y": Symbol("sdf"),
            "stress_disp_xx": Symbol("sdf"),
            "stress_disp_yy": Symbol("sdf"),
            "stress_disp_xy": Symbol("sdf"),
        },
        parameterization= param_ranges,
    )
    domain.add_constraint(hr_interior, "hr_interior")

    # INFERENCING THE VALUES - CASE INFERENCING
    invar_numpy = geo.sample_interior(
        100000,
        bounds={x: bounds_x, y: bounds_y},
        parameterization=inference_param_ranges,
    )
    point_cloud_inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_numpy,
        output_names={"u", "v", "sigma_xx", "sigma_yy", "sigma_xy"},
        batch_size=4096,
    )
    domain.add_inferencer(point_cloud_inference, "inf_data")

    slv = Solver(cfg, domain)

    slv.solve()

if __name__ == "__main__":
    run()