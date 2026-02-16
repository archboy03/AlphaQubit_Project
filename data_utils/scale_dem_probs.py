import stim

def scale_dem_probabilities(dem: stim.DetectorErrorModel, factor: float) -> stim.DetectorErrorModel:
    """
    Iterates through a Stim DEM and scales error probabilities by a factor.
    Recursively handles repeat blocks.
    """
    if factor == 1.0:
        return dem

    scaled_dem = stim.DetectorErrorModel()
    
    for instruction in dem:
        if isinstance(instruction, stim.DemRepeatBlock):
            # Recurse into the repeat block body to scale errors inside it
            scaled_body = scale_dem_probabilities(instruction.body_copy(), factor)
            scaled_dem.append("repeat", scaled_body, instruction.repeat_count)
        elif instruction.type == "error":
            original_p = instruction.args_copy()[0]
            new_p = max(0.0, min(original_p * factor, 1.0))
            scaled_dem.append(instruction.type, instruction.targets_copy(), new_p)
        else:
            # Pass through other instructions (detectors, logical_observables, shifts) unchanged
            scaled_dem.append(instruction)
            
    return scaled_dem