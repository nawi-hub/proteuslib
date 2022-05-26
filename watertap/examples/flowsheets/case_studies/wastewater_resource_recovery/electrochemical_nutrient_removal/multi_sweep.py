from watertap.tools.parameter_sweep import (
    _init_mpi,
    LinearSample,
    UniformSample,
    parameter_sweep,
)
import watertap.examples.flowsheets.case_studies.wastewater_resource_recovery.electrochemical_nutrient_removal.electrochemical_nutrient_removal as electrochemical_nutrient_removal


def set_up_sensitivity(m):
    outputs = {}
    optimize_kwargs = {"check_termination": False}
    opt_function = electrochemical_nutrient_removal.solve

    # create outputs
    outputs["LCOW"] = m.fs.costing.LCOW
    outputs["LCOS"] = m.fs.costing.LCOS

    return outputs, optimize_kwargs, opt_function


def run_analysis(case_num, nx, interpolate_nan_outputs=True):

    m = electrochemical_nutrient_removal.main()

    outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)

    sweep_params = {}
    if case_num == 1:
        # sensitivity analysis
        sweep_params["MgCl2_cost"] = LinearSample(
            m.fs.costing.magnesium_chloride_cost, 135, 538, nx
        )
    else:
        raise ValueError(f"{case_num} is not yet implemented")

    output_filename = "sensitivity_" + str(case_num)
    global_results = parameter_sweep(
        m,
        sweep_params,
        outputs,
        csv_results_file_name=output_filename,
        optimize_function=opt_function,
        optimize_kwargs=optimize_kwargs,
        interpolate_nan_outputs=interpolate_nan_outputs,
    )

    return global_results, sweep_params, m


if __name__ == "__main__":
    results, sweep_params, m = run_analysis(
        case_num=1, nx=11, interpolate_nan_outputs=False
    )
