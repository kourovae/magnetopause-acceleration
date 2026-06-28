"""
Magnetopause Acceleration Calculation
Based on Lin et al. (2010) magnetopause model and THEMIS spacecraft data
"""

import numpy as np
import math
from scipy.optimize import fsolve
from datetime import datetime
from astropy.time import Time
from geopack import geopack

# ============================================================
# Dipole tilt angle (PS) calculation using Geopack
# ============================================================

def calculate_ps_angle_direct(date_str, time_str):
    """
    Calculate the dipole tilt angle (PS) using Geopack recalc.
    
    Parameters:
    date_str : str
        Date in format 'YYYY-MM-DD'
    time_str : str
        Time in format 'HH:MM:SS'
    
    Returns:
    float : Dipole tilt angle PS in degrees
    """
    datetime_str = f"{date_str} {time_str}"
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    
    t = Time(dt)
    jd = t.jd
    unix_epoch_jd = 2440587.5
    ut_seconds = (jd - unix_epoch_jd) * 86400.0
    
    # recalc returns PSI in radians (dipole tilt angle)
    psi = geopack.recalc(ut_seconds)
    PS = np.degrees(psi)
    
    return PS

# ============================================================
# Coordinate transformation functions
# ============================================================

def cartesian_to_spherical_gsm(x, y, z):
    """Convert GSM Cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        return 0.0, 0.0, 0.0
    
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    if lon < 0:
        lon += 360
    
    return r, lat, lon

def spherical_to_cartesian_gsm(r, lat, lon):
    """Convert spherical coordinates to GSM Cartesian coordinates."""
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return np.array([x, y, z])

# ============================================================
# Lin et al. (2010) magnetopause model
# ============================================================

def lin2010_model(Bz, Btot, Pd, vLat, vLon, PS):
    """
    Magnetopause model from Lin et al., 2010 [JGR].
    
    Parameters:
    Bz   : IMF Bz component in GSM (nT)
    Btot : Total IMF magnitude (nT)
    Pd   : Solar wind dynamic pressure (nPa)
    vLat : GSM latitude (degrees)
    vLon : GSM longitude (degrees)
    PS   : Dipole tilt angle (degrees)
    
    Returns:
    float : Magnetopause standoff distance Rmp (Re)
    """
    # Input validation
    if Btot < 0 or Pd < 0 or Bz < -200:
        return 0.0
    
    PSr = np.radians(PS)
    
    # IMF magnetic pressure
    Pm = 3.978874e-04 * Btot**2
    
    # Model coefficients
    a0 = 12.544
    a1 = -0.194
    a2 = 0.305
    a3 = 0.0573
    a4 = 2.178
    a5 = 0.0571
    a6 = -0.999
    a7 = 16.473
    a8 = 0.00152
    a9 = 0.382
    a10 = 0.0431
    a11 = -0.00763
    a12 = -0.210
    a13 = 0.0405
    a14 = -4.430
    a15 = -0.636
    a16 = -2.600
    a17 = 0.832
    a18 = -5.328
    a19 = 1.103
    a20 = -0.907
    a21 = 1.450
    
    try:
        vla_rad = np.radians(vLat)
        vlo_rad = np.radians(vLon)
        
        # Zenith angle (te)
        te_cos = np.cos(vla_rad) * np.cos(vlo_rad)
        te_cos = np.clip(te_cos, -1.0, 1.0)
        te = np.arccos(te_cos)
        
        # Azimuthal angle (fi)
        fi_sq = (np.sin(vla_rad))**2 + (np.cos(vla_rad) * np.sin(vlo_rad))**2
        if fi_sq > 0:
            fi_cos = (np.cos(vla_rad) * np.sin(vlo_rad)) / np.sqrt(fi_sq)
            fi_cos = np.clip(fi_cos, -1.0, 1.0)
            fi = np.arccos(fi_cos)
            if np.sin(vla_rad) < 0:
                fi = -fi
        else:
            fi = 0.0
        
        # Model parameters
        en = a21
        es = a21
        tn = a19 + a20 * PSr
        ts = a19 - a20 * PSr
        
        xn = np.arccos(np.cos(te) * np.cos(tn) + np.sin(te) * np.sin(tn) * np.cos(fi - np.pi/2.0))
        xs = np.arccos(np.cos(te) * np.cos(ts) + np.sin(te) * np.sin(ts) * np.cos(fi - 3.0 * np.pi / 2.0))
        
        # Coefficients
        b0 = a6 + a7 * (np.exp(a8 * Bz) - 1.0) / (np.exp(a9 * Bz) + 1.0)
        b1 = a10
        b2 = a11 + a12 * PSr
        b3 = a13
        
        cn = a14 * (Pd + Pm)**a15
        cs = cn
        
        dn = a16 + a17 * PSr + a18 * PSr**2
        ds = a16 - a17 * PSr + a18 * PSr**2
        
        # Shape function f
        f_term = np.cos(te / 2.0) + a5 * np.sin(2.0 * te) * (1.0 - np.exp(-te))
        f_exponent = b0 + b1 * np.cos(fi) + b2 * np.sin(fi) + b3 * (np.sin(fi))**2
        f = f_term ** f_exponent
        
        # Standoff distance R0
        R0_term = 1.0 + a2 * (np.exp(a3 * Bz) - 1.0) / (np.exp(a4 * Bz) + 1.0)
        R0 = R0_term * a0 * (Pd + Pm)**a1
        
        # Final magnetopause distance
        Rmp = R0 * f + cn * np.exp(dn * xn**en) + cs * np.exp(ds * xs**es)
        
        return Rmp
        
    except Exception as e:
        print(f"Error in Lin2010 model: {e}")
        return 0.0

# ============================================================
# Pressure determination from satellite position
# ============================================================

def find_pressure_for_satellite(x, y, z, Bz=0.0, Btot=5.0, PS=0.0):
    """
    Find the solar wind dynamic pressure Pd such that the model
    magnetopause passes through the satellite position.
    """
    r, lat, lon = cartesian_to_spherical_gsm(x, y, z)
    
    def equation(Pd):
        R_mp = lin2010_model(Bz, Btot, Pd, lat, lon, PS)
        return R_mp - r
    
    try:
        Pd_sol = fsolve(equation, 1.0, full_output=True)
        if Pd_sol[2] == 1:
            Pd_found = Pd_sol[0][0]
            if Pd_found > 0:
                R_mp_final = lin2010_model(Bz, Btot, Pd_found, lat, lon, PS)
                return Pd_found, R_mp_final, lat, lon
        
        return None, None, lat, lon
            
    except Exception as e:
        print(f"Error solving for pressure: {e}")
        return None, None, lat, lon

# ============================================================
# Magnetopause normal computation
# ============================================================

def compute_normal_direct(lat, lon, Pd, Bz, Btot, PS, delta=0.001):
    """
    Compute the outward normal to the magnetopause using finite differences.
    """
    r0 = lin2010_model(Bz, Btot, Pd, lat, lon, PS)
    point0 = spherical_to_cartesian_gsm(r0, lat, lon)
    
    # Variations in latitude and longitude
    point_lat_plus = spherical_to_cartesian_gsm(
        lin2010_model(Bz, Btot, Pd, lat + delta, lon, PS), lat + delta, lon)
    point_lat_minus = spherical_to_cartesian_gsm(
        lin2010_model(Bz, Btot, Pd, lat - delta, lon, PS), lat - delta, lon)
    
    point_lon_plus = spherical_to_cartesian_gsm(
        lin2010_model(Bz, Btot, Pd, lat, lon + delta, PS), lat, lon + delta)
    point_lon_minus = spherical_to_cartesian_gsm(
        lin2010_model(Bz, Btot, Pd, lat, lon - delta, PS), lat, lon - delta)
    
    tangent_lat = (point_lat_plus - point_lat_minus) / (2.0 * delta)
    tangent_lon = (point_lon_plus - point_lon_minus) / (2.0 * delta)
    
    normal_gsm = np.cross(tangent_lat, tangent_lon)
    norm = np.linalg.norm(normal_gsm)
    
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    
    normal_gsm = normal_gsm / norm
    if np.dot(normal_gsm, point0) < 0:
        normal_gsm = -normal_gsm
    
    return normal_gsm

def spherical_normal_to_gsm(normal_sph, lat, lon):
    """
    Convert spherical normal vector to GSM Cartesian coordinates.
    """
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    theta = np.pi / 2.0 - lat_rad
    
    T = np.array([
        [np.sin(theta) * np.cos(lon_rad), np.cos(theta) * np.cos(lon_rad), -np.sin(lon_rad)],
        [np.sin(theta) * np.sin(lon_rad), np.cos(theta) * np.sin(lon_rad), np.cos(lon_rad)],
        [np.cos(theta), -np.sin(theta), 0]
    ])
    
    normal_gsm = T @ normal_sph
    norm = np.linalg.norm(normal_gsm)
    
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    
    return normal_gsm / norm

# ============================================================
# Intersection search along normal
# ============================================================

def find_intersection_along_normal(start_point, normal_gsm, Pd, Bz, Btot, PS,
                                   max_distance=10.0, step=0.01):
    """
    Find the intersection of the normal line with the magnetopause
    for a given solar wind dynamic pressure Pd.
    """
    directions = [1.0, -1.0]
    
    for direction in directions:
        distance = 0.0
        
        while distance <= max_distance:
            current_point = start_point + normal_gsm * distance * direction
            
            r_current, lat_current, lon_current = cartesian_to_spherical_gsm(*current_point)
            
            r_mp_target = lin2010_model(Bz, Btot, Pd, lat_current, lon_current, PS)
            
            diff = np.linalg.norm(current_point) - r_mp_target
            
            if abs(diff) < 0.01:
                return current_point, distance * direction
            
            distance += step
    
    return None, None

# ============================================================
# Velocity calculations
# ============================================================

def calculate_mp_velocity_compression(sat_outer_coords, sat_inner_coords,
                                      t_outer, t_inner, Bz=0.0, Btot=5.0, PS=0.0):
    """
    Calculate magnetopause velocity during compression (inward motion).
    """
    print("=" * 70)
    print("MAGNETOPAUSE COMPRESSION")
    print("=" * 70)
    
    Pd_outer, R_outer, lat_outer, lon_outer = find_pressure_for_satellite(
        *sat_outer_coords, Bz, Btot, PS)
    Pd_inner, R_inner, lat_inner, lon_inner = find_pressure_for_satellite(
        *sat_inner_coords, Bz, Btot, PS)
    
    if Pd_outer is None or Pd_inner is None:
        print("Error: could not determine pressures")
        return None
    
    print(f"Pressures:")
    print(f"  Pd_outer (t1): {Pd_outer:.3f} nPa")
    print(f"  Pd_inner (t2): {Pd_inner:.3f} nPa")
    
    S_outer = np.array(sat_outer_coords)
    S_inner = np.array(sat_inner_coords)
    
    normal_sph = compute_normal_direct(lat_outer, lon_outer, Pd_outer, Bz, Btot, PS)
    normal_gsm = spherical_normal_to_gsm(normal_sph, lat_outer, lon_outer)
    
    intersection_mp2, distance_to_mp2 = find_intersection_along_normal(
        S_outer, normal_gsm, Pd_inner, Bz, Btot, PS)
    
    if intersection_mp2 is None:
        print("Error: intersection not found")
        return None
    
    dt = t_inner - t_outer
    if dt <= 0:
        print("Error: invalid time interval")
        return None
    
    mp_travel_distance = abs(distance_to_mp2)
    v_normal = mp_travel_distance / dt
    v_normal_kms = v_normal * 6371.0
    
    print(f"\nRESULTS (COMPRESSION):")
    print(f"  Travel distance: {mp_travel_distance:.3f} Re")
    print(f"  Time interval: {dt:.1f} s")
    print(f"  Velocity: {v_normal:.4f} Re/s = {v_normal_kms:.2f} km/s")
    print(f"  Direction: inward (toward Earth)")
    
    return {
        'v_normal_re_s': v_normal,
        'v_normal_km_s': v_normal_kms,
        'mp_travel_distance': mp_travel_distance,
        'dt': dt,
        'Pd_outer': Pd_outer,
        'Pd_inner': Pd_inner,
        'S_outer': S_outer,
        'S_inner': S_inner,
        'intersection_mp2': intersection_mp2,
        'normal_gsm': normal_gsm,
        'movement_direction': 'inward'
    }

def calculate_mp_velocity_expansion(sat_inner_coords, sat_outer_coords,
                                    t_inner, t_outer, Bz=0.0, Btot=5.0, PS=0.0):
    """
    Calculate magnetopause velocity during expansion (outward motion).
    """
    print("=" * 70)
    print("MAGNETOPAUSE EXPANSION")
    print("=" * 70)
    
    Pd_inner, R_inner, lat_inner, lon_inner = find_pressure_for_satellite(
        *sat_inner_coords, Bz, Btot, PS)
    Pd_outer, R_outer, lat_outer, lon_outer = find_pressure_for_satellite(
        *sat_outer_coords, Bz, Btot, PS)
    
    if Pd_inner is None or Pd_outer is None:
        print("Error: could not determine pressures")
        return None
    
    print(f"Pressures:")
    print(f"  Pd_inner (t1): {Pd_inner:.3f} nPa")
    print(f"  Pd_outer (t2): {Pd_outer:.3f} nPa")
    
    S_inner = np.array(sat_inner_coords)
    S_outer = np.array(sat_outer_coords)
    
    normal_sph = compute_normal_direct(lat_inner, lon_inner, Pd_inner, Bz, Btot, PS)
    normal_gsm = spherical_normal_to_gsm(normal_sph, lat_inner, lon_inner)
    
    intersection_mp2, distance_to_mp2 = find_intersection_along_normal(
        S_inner, normal_gsm, Pd_outer, Bz, Btot, PS)
    
    if intersection_mp2 is None:
        print("Error: intersection not found")
        return None
    
    dt = t_outer - t_inner
    if dt <= 0:
        print("Error: invalid time interval")
        return None
    
    mp_travel_distance = abs(distance_to_mp2)
    v_normal = mp_travel_distance / dt
    v_normal_kms = v_normal * 6371.0
    
    print(f"\nRESULTS (EXPANSION):")
    print(f"  Travel distance: {mp_travel_distance:.3f} Re")
    print(f"  Time interval: {dt:.1f} s")
    print(f"  Velocity: {v_normal:.4f} Re/s = {v_normal_kms:.2f} km/s")
    print(f"  Direction: outward (away from Earth)")
    
    return {
        'v_normal_re_s': v_normal,
        'v_normal_km_s': v_normal_kms,
        'mp_travel_distance': mp_travel_distance,
        'dt': dt,
        'Pd_inner': Pd_inner,
        'Pd_outer': Pd_outer,
        'S_inner': S_inner,
        'S_outer': S_outer,
        'intersection_mp2': intersection_mp2,
        'normal_gsm': normal_gsm,
        'normal_sph': normal_sph,
        'movement_direction': 'outward'
    }

# ============================================================
# Acceleration calculation from three satellites
# ============================================================

def calculate_mp_acceleration_three_satellites(satellite_data_list, Bz, Btot, PS):
    """
    Calculate magnetopause acceleration using three successive crossings.
    
    Parameters:
    satellite_data_list : list of tuples
        [(coords1, t1), (coords2, t2), (coords3, t3)]
        where coords = (x, y, z) in Re, t = absolute time in seconds
    """
    if len(satellite_data_list) != 3:
        print("Error: exactly 3 satellites required")
        return None
    
    sorted_data = sorted(satellite_data_list, key=lambda x: x[1])
    
    print("=" * 70)
    print("MAGNETOPAUSE ACCELERATION FROM THREE SATELLITES")
    print("=" * 70)
    
    (coords1, t1), (coords2, t2), (coords3, t3) = sorted_data
    
    print(f"Satellite 1: {coords1} Re, t = {t1} s")
    print(f"Satellite 2: {coords2} Re, t = {t2} s")
    print(f"Satellite 3: {coords3} Re, t = {t3} s")
    
    # Velocity 1 (between satellites 1 and 2)
    print("\n--- VELOCITY 1 (t1 → t2) ---")
    if t1 < t2:
        r1 = np.linalg.norm(coords1)
        r2 = np.linalg.norm(coords2)
        if r1 > r2:
            velocity_result1 = calculate_mp_velocity_compression(
                coords1, coords2, t1, t2, Bz, Btot, PS)
        else:
            velocity_result1 = calculate_mp_velocity_expansion(
                coords1, coords2, t1, t2, Bz, Btot, PS)
    else:
        print("Error: invalid times t1 and t2")
        return None
    
    # Velocity 2 (between satellites 2 and 3)
    print("\n--- VELOCITY 2 (t2 → t3) ---")
    if t2 < t3:
        r2 = np.linalg.norm(coords2)
        r3 = np.linalg.norm(coords3)
        if r2 > r3:
            velocity_result2 = calculate_mp_velocity_compression(
                coords2, coords3, t2, t3, Bz, Btot, PS)
        else:
            velocity_result2 = calculate_mp_velocity_expansion(
                coords2, coords3, t2, t3, Bz, Btot, PS)
    else:
        print("Error: invalid times t2 and t3")
        return None
    
    if not velocity_result1 or not velocity_result2:
        print("Error: velocity calculation failed")
        return None
    
    # Acceleration
    print("\n--- ACCELERATION ---")
    
    time1 = (t1 + t2) / 2.0
    time2 = (t2 + t3) / 2.0
    time_interval = time2 - time1
    
    if time_interval <= 0:
        print("Error: invalid time interval for acceleration")
        return None
    
    v1 = velocity_result1['v_normal_km_s']
    v2 = velocity_result2['v_normal_km_s']
    delta_velocity = v2 - v1
    acceleration = delta_velocity / time_interval
    
    print(f"Velocity 1: {v1:.3f} km/s (t ~ {time1:.1f} s)")
    print(f"Velocity 2: {v2:.3f} km/s (t ~ {time2:.1f} s)")
    print(f"Velocity change: {delta_velocity:.3f} km/s")
    print(f"Time interval: {time_interval:.1f} s")
    print(f"Acceleration: {acceleration:.6f} km/s²")
    
    if acceleration > 0.001:
        print("✓ ACCELERATION")
    elif acceleration < -0.001:
        print("✓ DECELERATION")
    else:
        print("✓ CONSTANT VELOCITY")
    
    return {
        'velocity1_km_s': v1,
        'velocity2_km_s': v2,
        'acceleration_km_s2': acceleration,
        'acceleration_m_s2': acceleration * 1000.0,
        'time_interval_acceleration': time_interval,
        'time1': time1,
        'time2': time2,
        'movement_direction': velocity_result1['movement_direction']
    }

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example: calculate dipole tilt angle for a given date/time
    date = '2007-08-29'
    time = '15:40:00'
    PS = calculate_ps_angle_direct(date, time)
    print(f"Dipole tilt angle PS: {PS:.3f} degrees")
    
    # Example parameters
    Bz, Btot = 2.5, 3.0
    
    # Example: three satellite crossings for compression
    satellite_data = [
        ((10.279, -2.031, -2.100), 0),
        ((10.118, -1.984, -2.045), 13),
        ((10.026, -1.853, -2.064), 19)
    ]
    
    acceleration_result = calculate_mp_acceleration_three_satellites(
        satellite_data, Bz, Btot, PS
    )