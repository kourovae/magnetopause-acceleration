# Magnetopause Acceleration Calculation

Python code for calculating magnetopause velocity and acceleration from successive crossings by THEMIS spacecraft, based on the Lin et al. (2010) magnetopause model.

## Description

This code implements:
- Lin et al. (2010) magnetopause model
- Dipole tilt angle (PS) calculation using Geopack
- Determination of solar wind dynamic pressure from satellite crossings
- Calculation of magnetopause normal vectors
- Velocity and acceleration computation for magnetopause expansion and compression

## Requirements

- Python 3.13.7
- NumPy (2.2.4)
- SciPy (1.16.1)
- Matplotlib (3.10.1)
- Astropy (7.1.0)
- Geopack (1.0.12)

## Installation

```bash
pip install -r requirements.txt

## Usage

from magnetopause_acceleration import calculate_ps_angle_direct, calculate_mp_acceleration_three_satellites

# Calculate dipole tilt angle
PS = calculate_ps_angle_direct('2007-08-29', '15:40:00')

# Example satellite data (three crossings)
satellite_data = [
    ((10.279, -2.031, -2.100), 0),
    ((10.118, -1.984, -2.045), 13),
    ((10.026, -1.853, -2.064), 19)
]

# Calculate acceleration
acceleration = calculate_mp_acceleration_three_satellites(
    satellite_data, Bz=2.5, Btot=3.0, PS=PS
)

## License

MIT License

## Citation

If you use this code, please cite:

Kourova E.A., Dmitriev A.V., Mishin V.V. (2026). Code for magnetopause acceleration calculation and figure generation [Software]. Zenodo. https://doi.org/10.5281/zenodo.20999491
