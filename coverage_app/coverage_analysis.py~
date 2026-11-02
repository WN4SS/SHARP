"""
coverage_analysis.py

This module includes methods for performing a coverage analysis on a scenario.
"""

import os
import csv
import math
import random
import re
import shutil
import json
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import geopandas as gpd

from math import sin, cos, radians, degrees
from shapely.geometry import Point, Polygon, box, LineString as ShapelyLineString
from shapely.strtree import STRtree
from .path_loss import compute_path_loss, knife_edge_loss_dB, compute_acir_db
from .path_loss import directional_antenna_gain_db
from .plotting import plot_best_rsrp_heatmap, plot_best_sinr_heatmap
from .plotting import plot_best_throughput_heatmap
from .plotting import plot_grid_based_sinr_heatmap
from .plotting import plot_measurement_points_3d

k_b = 1.38e-23 # Boltzmann constant in J/K
T = 290        # Temperature in K
B = 10e6       # Default channel bandwidth in Hz, when an antenna declares none
Grx = 7        # Receiver gain in dBi


def thermal_noise_dbm(bandwidth_hz=B):
    """Thermal noise floor kTB over *bandwidth_hz*, in dBm."""
    return 10 * math.log10(k_b * T * bandwidth_hz * 1000)


noise_power_dBm = thermal_noise_dbm(B)  # ~ -104 dBm over 10 MHz

LTE_RB_BY_BANDWIDTH_HZ = {
    1.4e6: 6,
    3e6: 15,
    5e6: 25,
    10e6: 50,
    15e6: 75,
    20e6: 100,
}

OSM_MAP_API_URL = "https://api.openstreetmap.org/api/0.6/map"
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
OSM_USER_AGENT = "CoverageApp/1.0"
OSM_BBOX_PADDING_DEG = 0.0005
BUILDING_CACHE_DIRNAME = ".coverage_cache"

# GlobalBuildingAtlas WFS endpoint
GBA_WFS_URL = "https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/ows"
GBA_WFS_TYPENAME = "global3D:lod1_global"
GBA_WFS_TIMEOUT = 30
DEFAULT_OSM_LEVEL_HEIGHT_M = 3.0
OSM_LEVEL_HEIGHT_BY_BUILDING_TAG = {
    "apartments": 3.1,
    "commercial": 3.5,
    "detached": 3.0,
    "dormitory": 3.1,
    "garage": 2.8,
    "hangar": 6.0,
    "hospital": 3.5,
    "hotel": 3.2,
    "house": 3.0,
    "industrial": 4.5,
    "office": 3.5,
    "parking": 2.8,
    "residential": 3.0,
    "retail": 4.0,
    "school": 3.5,
    "supermarket": 4.0,
    "terrace": 3.0,
    "university": 3.5,
    "warehouse": 4.5,
}

def dBm_to_mW(dBm):
        return 10 ** (dBm / 10)

def mW_to_dBm(mW):
        return 10 * math.log10(mW)

def get_num_resource_blocks(bandwidth_hz=B):
        for nominal_bw_hz, num_rbs in LTE_RB_BY_BANDWIDTH_HZ.items():
                if math.isclose(bandwidth_hz, nominal_bw_hz, rel_tol=0.0, abs_tol=1e5):
                        return num_rbs
        return max(int(round(bandwidth_hz / 180e3)), 1)

def approximate_lte_rsrp_rsrq(received_power_dbm, rssi_mW, bandwidth_hz=B):
    """
    Approximate LTE-style RSRP/RSRQ from the modeled wideband received power.

    The propagation engine computes wideband serving/interfering powers. To export
    metrics with LTE-like semantics, we convert the serving wideband power into a
    per-resource-element RSRP using 12 subcarriers per RB and the RB count implied
    by the measurement bandwidth. RSRQ is then derived as N_RB * RSRP / RSSI.
    """
    if received_power_dbm <= -999 or rssi_mW <= 0:
            return -999.0, -999.0

    num_rbs = get_num_resource_blocks(bandwidth_hz)
    received_power_mW = dBm_to_mW(received_power_dbm)
    rsrp_mW = received_power_mW / (12 * num_rbs)
    rsrp_dbm = mW_to_dBm(rsrp_mW)
    rsrq_linear = (num_rbs * rsrp_mW) / rssi_mW
    rsrq_db = 10 * math.log10(rsrq_linear) if rsrq_linear > 0 else -999.0
    return rsrp_dbm, rsrq_db

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns the 2D ground distance in meters using the Haversine formula.
    """
    R = 6371000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lmb = radians(lon2 - lon1)
    a = (sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lmb / 2) ** 2)
    c_val = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c_val

def calculate_azimuth(lat1, lon1, lat2, lon2):
    """
    Computes azimuth angle between two points in degrees [0..360).
    """
    lat1_r = radians(lat1); lon1_r = radians(lon1)
    lat2_r = radians(lat2); lon2_r = radians(lon2)
    delta_lon = lon2_r - lon1_r
    y = sin(delta_lon) * cos(lat2_r)
    x = cos(lat1_r) * sin(lat2_r) - sin(lat1_r) * cos(lat2_r) * cos(delta_lon)
    return (degrees(math.atan2(y, x)) + 360) % 360

def _parse_numeric_value(raw_value):
    """
    Extract the first numeric token from an OSM tag value.
    """
    if raw_value is None:
            return None
    text = str(raw_value).strip().lower().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
            return None
    return float(match.group(0))

def _parse_height_meters(raw_value):
    """
    Parse a height tag, converting feet to meters when needed.
    """
    value = _parse_numeric_value(raw_value)
    if value is None:
        return None
    text = str(raw_value).strip().lower()
    if "ft" in text or "feet" in text or "foot" in text:
            return value * 0.3048
    return value

def _get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def _get_building_cache_root(cache_root=None):
    root = cache_root or os.path.join(_get_project_root(), BUILDING_CACHE_DIRNAME, "buildings")
    os.makedirs(root, exist_ok=True)
    return root

def _resolve_aoi_geometry(polygons_gdf=None, points=None):
    if polygons_gdf is not None and len(polygons_gdf) > 0:
        local_gdf = polygons_gdf
        if local_gdf.crs is None:
            local_gdf = local_gdf.set_crs("EPSG:4326")
        elif local_gdf.crs.to_string() != "EPSG:4326":
            local_gdf = local_gdf.to_crs("EPSG:4326")
            
        aoi_geometry = local_gdf.union_all()
        if not aoi_geometry.is_empty:
            return aoi_geometry

    if points:
        lats = [point["lat"] for point in points]
        lons = [point["lon"] for point in points]
        return box(min(lons), min(lats), max(lons), max(lats))

    return None

def generate_points_in_polygon(polygon, n):
    """
    generates n random points strictly inside a polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        rx = random.uniform(minx, maxx)
        ry = random.uniform(miny, maxy)
        candidate = Point(rx, ry)
        if polygon.contains(candidate):
            points.append({"lat": candidate.y, "lon": candidate.x})
            attempts += 1
    return points

def generate_points_in_circle(center_lat, center_lon, radius_m, num_points):
    """
        Generates random points in a circle of radius_m around (center_lat, center_lon).
        """
    pts = []
    for _ in range(num_points):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(0, radius_m)
        dlat = (dist * cos(angle)) / 111320
        dlon = (dist * sin(angle)) / (111320 * cos(radians(center_lat)))
        pts.append({"lat": center_lat + dlat, "lon": center_lon + dlon})
    return pts

def antenna_gain(point_azimuth, antenna_azimuth, Gtx, vertical_angle_deg=0.0,
                 downtilt_deg=0.0):
    """
    Absolute directional antenna gain (dBi) from the measured radiation
    patterns (``coverage_app/patterns/*.pat``), via the shared 3GPP H+V model
    in :mod:`coverage_app.path_loss`.

    ``Gtx`` is the peak (boresight) gain; ``vertical_angle_deg`` is the signed
    geometric elevation from the antenna to the point; ``downtilt_deg`` is the
    antenna's mechanical downtilt (added to the elevation, default 0).
    """
    az_off = point_azimuth - antenna_azimuth
    el_off = vertical_angle_deg + downtilt_deg
    return directional_antenna_gain_db(Gtx, az_off, el_off)

def process_metrics(antennas, points, output_folder, path_loss_model='fspl', base_height=None,
                    mobile_height=None, alpha=123.2, beta=1.2, n=4.1, shadowing=False, omni=True,
                    environment='urban',
                    all_antennas_for_aci=None, aclr_db=45.0, acs_db=33.0, channel_bw_mhz=10.0,
                    scenario='uma', diffraction_losses=None, point_heights=None, antenna_heights=None):
    """
    Main coverage computation for a set of antennas:
    - RSRP, RSRQ, SINR, path-loss, etc.
    - When *all_antennas_for_aci* is provided (list of all antennas across all
      frequencies), adjacent-channel interference (ACI) is added to the SINR
          denominator following 3GPP TR 36.942 §4.2.3:
              1/ACIR(Δf) = 1/ACLR(Δf) + 1/ACS
              ACLR(n)    = aclr_db + 20*(n-1) dB   [n = |Δf|/BW, first-order roll-off]
          Default values: ACLR₁=45 dB (3GPP TS 36.104 Table 6.6.1-1),
                          ACS=33 dB   (3GPP TS 36.101 Table 7.5-1).
        - Saves per-antenna CSV + best coverage CSV (named 'best_data.csv' by default).
        """
    metrics = {
        i: {"distances": [], "SINR": [], "PathLoss": [], "RSRP": [], "RSRQ": [],
            "RSSI": []}
                for i in range(len(antennas))
    }

    best_data = {
        "Latitude": [],
        "Longitude": [],
        "Measurement Point Height (m)": [],
        "Best Antenna Distance (m)": [],
        "Best Antenna Index": [],
        "Best Antenna Path Loss (dB)": [],
        "Best Antenna RSRP (dBm)": [],
        "Best Antenna RSRQ (dB)": [],
        "Best Antenna SINR (dB)": [],
        "Best Antenna RSSI (dBm)": [],
        "Best Antenna Bandwidth (MHz)": []
    }

    os.makedirs(output_folder, exist_ok=True)

    for point_index, point in enumerate(points):
        lat, lon = point["lat"], point["lon"]
        point_height = point_heights[point_index]
        for i, antenna in enumerate(antennas):
            dist_2d = haversine_distance(antenna["lat"], antenna["lon"], lat, lon)
            antenna_height = antenna_heights[i]
            height_diff = (antenna_height - point_height)
            dist_3d = math.sqrt(dist_2d ** 2 + height_diff ** 2)

            # Compute path-loss
            pl = compute_path_loss(distance_m=dist_3d,
                                   model=path_loss_model,
                                   distance_2d_m=dist_2d,
                                   frequency_mhz=antenna["frequency"],
                                   base_height=antenna_height,
                                   mobile_height=point_height,
                                   alpha=alpha,
                                   beta=beta,
                                   n=n,
                                   shadowing=shadowing,
                                   omni=omni,
                                   environment=environment,
                                   scenario=scenario)

            # Compute azimuth (horizontal)
            az = calculate_azimuth(antenna["lat"], antenna["lon"], lat, lon)

            # Elevation (vertical) angle — signed: positive above boresight
            if dist_2d > 0:
                v_ang_deg = degrees(math.atan2(height_diff, dist_2d))
            else:
                v_ang_deg = 0.0

            # 3D gain from the measured H+V radiation patterns
            gain = antenna_gain(az, antenna["azimuth"], antenna.get("Gtx", 19.5),
                                vertical_angle_deg=v_ang_deg,
                                downtilt_deg=antenna.get("downtilt", 0.0))

            ptx = antenna.get("Ptx", 47)
            # Bandwidth belongs to the receiver: a point is measured over the
            # channel of the antenna serving it. Falls back to the call-level
            # default when the scenario declares none.
            bw_mhz = antenna.get("bandwidth_mhz")
            bw_mhz = float(channel_bw_mhz) if bw_mhz is None or pd.isna(bw_mhz) else float(bw_mhz)
            bw_hz = bw_mhz * 1e6
            diff_loss = diffraction_losses[point_index]
            received_power_dbm = ptx + gain + Grx - pl - diff_loss

            # Co-channel interference (only from antennas of the same frequency)
            main_freq = antenna["frequency"]
            total_interf_mW = 0.0
            for j, interf in enumerate(antennas):
                    if j != i and interf["frequency"] == main_freq:
                        dist_i = haversine_distance(interf["lat"], interf["lon"], lat, lon)
                        az_i = calculate_azimuth(interf["lat"], interf["lon"], lat, lon)

                        interf_height = antenna_heights[j]
                        height_diff_i = (interf_height - point_height)
                        dist_3d_i = math.sqrt(dist_i ** 2 + height_diff_i ** 2)
                        if dist_i > 0:
                            v_ang_deg_i = degrees(math.atan2(height_diff_i, dist_i))
                        else:
                            v_ang_deg_i = 0.0

                        gain_i = antenna_gain(az_i,
                                              interf["azimuth"],
                                              interf.get("Gtx", 19.5),
                                              vertical_angle_deg=v_ang_deg_i,
                                              downtilt_deg=interf.get("downtilt", 0.0))
                        ptx_i = interf.get("Ptx", 47)

                        pl_i = compute_path_loss(distance_m=dist_3d_i,
                                                 distance_2d_m=dist_i,
                                                 frequency_mhz=interf["frequency"],
                                                 model=path_loss_model,
                                                 base_height=interf_height,
                                                 mobile_height=point_height,
                                                 alpha=alpha,
                                                 beta=beta,
                                                 n=n,
                                                 shadowing=shadowing,
                                                 omni=omni,
                                                 environment=environment,
                                                 scenario=scenario)
                        # Diffraction also attenuates interfering signals
                        diff_loss_i = diffraction_losses[point_index]
                        interf_power_dbm = ptx_i + gain_i + Grx - pl_i - diff_loss_i
                        total_interf_mW += dBm_to_mW(interf_power_dbm)

            # Adjacent-channel interference (ACI) — 3GPP TR 36.942 §4.2.3
            # SINR = S / (I_CCI + I_ACI + N)
            # I_ACI = sum_j  P_j / ACIR(|f_j - f_i|)
            # ACIR  = 1 / (1/ACLR(Δf) + 1/ACS)
            # ACLR(n) = ACLR_1 + 20*(n-1) dB  [first-order filter roll-off]
            if all_antennas_for_aci is not None:
                for aci_ant in all_antennas_for_aci:
                    if aci_ant["frequency"] == main_freq:
                        continue  # co-channel: already counted above

                    delta_f = abs(aci_ant["frequency"] - main_freq)
                    acir_db_val = compute_acir_db(delta_f, bw_mhz, aclr_db, acs_db)
                    # Skip when attenuation is so high that ACI is negligible
                    if acir_db_val > 80.0:
                        continue

                    dist_aci = haversine_distance(aci_ant["lat"], aci_ant["lon"], lat, lon)
                    az_aci   = calculate_azimuth(aci_ant["lat"], aci_ant["lon"], lat, lon)
                    aci_h    = aci_ant.get("height", base_height if base_height is not None else 30)
                    hdiff_aci = aci_h - point_height
                    dist_3d_aci = math.sqrt(dist_aci ** 2 + hdiff_aci ** 2)
                    vang_aci = degrees(math.atan2(hdiff_aci, dist_aci)) if dist_aci > 0 else 0.0

                    gain_aci = antenna_gain(az_aci,
                                            aci_ant["azimuth"],
                                            aci_ant.get("Gtx", 19.5),
                                            vertical_angle_deg=vang_aci,
                                            downtilt_deg=aci_ant.get("downtilt", 0.0))
                    ptx_aci = aci_ant.get("Ptx", 47)
                    pl_aci  = compute_path_loss(distance_m=dist_3d_aci,
                                                distance_2d_m=dist_aci,
                                                frequency_mhz=aci_ant["frequency"],
                                                model=path_loss_model,
                                                base_height=aci_h,
                                                mobile_height=point_height,
                                                alpha=alpha,
                                                beta=beta,
                                                n=n,
                                                shadowing=shadowing,
                                                omni=omni,
                                                environment=environment,
                                                scenario=scenario)
                    diff_loss_aci = diffraction_losses[point_index]
                    aci_power_dbm = ptx_aci + gain_aci + Grx - pl_aci - diff_loss_aci
                    # Apply ACIR suppression (linear division)
                    total_interf_mW += dBm_to_mW(aci_power_dbm) / (10.0 ** (acir_db_val / 10.0))

            noise_mW = dBm_to_mW(thermal_noise_dbm(bw_hz))
            numerator_mW = dBm_to_mW(received_power_dbm) if received_power_dbm > -999 else 0.0
            denominator_mW = total_interf_mW + noise_mW
            sinr_mW = numerator_mW / denominator_mW if denominator_mW > 0 else 0.0
            sinr_val = 10 * math.log10(sinr_mW) if sinr_mW > 0 else -float('inf')

            rssi_mW = numerator_mW + total_interf_mW + noise_mW
            rsrp, rsrq = approximate_lte_rsrp_rsrq(received_power_dbm, rssi_mW,
                                                   bandwidth_hz=bw_hz)

            metrics[i]["distances"].append(dist_3d)
            metrics[i]["SINR"].append(sinr_val)
            metrics[i]["PathLoss"].append(pl)
            metrics[i]["RSRP"].append(rsrp)
            metrics[i]["RSRQ"].append(rsrq)
            metrics[i]["RSSI"].append(mW_to_dBm(rssi_mW) if rssi_mW > 0 else -999.0)

            best_sinr = -float('inf')
            best_metrics_for_point = {}

            # Track best (highest SINR) for this point
            if sinr_val > best_sinr:
                best_sinr = sinr_val
                best_metrics_for_point = {
                    "distance": dist_3d,
                    "antenna_index": i + 1,
                    "path_loss": pl,
                    "rsrp": rsrp,
                    "rsrq": rsrq,
                    "sinr": sinr_val,
                    "rssi": mW_to_dBm(rssi_mW) if rssi_mW > 0 else -999.0,
                    "bandwidth_mhz": bw_mhz
                }

        # If no valid best => dummy
        if best_sinr == -float('inf'):
            best_metrics_for_point = {
                "distance": 999999.0,
                "antenna_index": -1,
                "path_loss": 999.0,
                "rsrp": -999.0,
                "rsrq": -999.0,
                "sinr": -999.0,
                "rssi": -999.0,
                "bandwidth_mhz": float(channel_bw_mhz)
            }

        best_data["Latitude"].append(lat)
        best_data["Longitude"].append(lon)
        best_data["Measurement Point Height (m)"].append(point_height)
        best_data["Best Antenna Distance (m)"].append(best_metrics_for_point["distance"])
        best_data["Best Antenna Index"].append(f"Antenna {best_metrics_for_point['antenna_index']}")
        best_data["Best Antenna Path Loss (dB)"].append(best_metrics_for_point["path_loss"])
        best_data["Best Antenna RSRP (dBm)"].append(best_metrics_for_point["rsrp"])
        best_data["Best Antenna RSRQ (dB)"].append(best_metrics_for_point["rsrq"])
        best_data["Best Antenna SINR (dB)"].append(best_metrics_for_point["sinr"])
        best_data["Best Antenna RSSI (dBm)"].append(best_metrics_for_point["rssi"])
        best_data["Best Antenna Bandwidth (MHz)"].append(best_metrics_for_point["bandwidth_mhz"])

    # Save per-antenna metrics
    antennas_data_folder = os.path.join(output_folder, "antennas_data")
    os.makedirs(antennas_data_folder, exist_ok=True)
    for i, ant_metric in metrics.items():
        df_ant = pd.DataFrame({
            "Latitude": [p["lat"] for p in points],
            "Longitude": [p["lon"] for p in points],
            "Measurement Point Height (m)": [
                p.get("height") if p.get("height") is not None else (mobile_height if mobile_height is not None else 0.0)
                            for p in points
            ],
            "Measurement Point Height Source": [
                p.get("height_source", "point_height") if p.get("height") is not None
                            else ("global_mobile_height" if mobile_height is not None else "default_0m")
                            for p in points
            ],
            "Distance (m)": ant_metric["distances"],
            "PathLoss (dB)": ant_metric["PathLoss"],
            "SINR (dB)": ant_metric["SINR"],
            "RSRP (dBm)": ant_metric["RSRP"],
            "RSRQ (dB)": ant_metric["RSRQ"],
            "Antenna Index": [i + 1] * len(points)
        })
        df_ant.to_csv(os.path.join(antennas_data_folder, f"antenna_{i+1}_metrics.csv"), index=False)

    # Save best coverage data
    best_df = pd.DataFrame(best_data)
    best_df.to_csv(os.path.join(output_folder, "best_data.csv"), index=False)

def combine_frequency_best_data(freq_best_data_paths, output_csv):
    """
    Given multiple CSVs (one per frequency) merges them into a single best_data.csv,
    picking the highest SINR at each point across frequencies.
    """
    if not freq_best_data_paths:
        return

    # Read the first CSV as base
    combined_df = pd.read_csv(freq_best_data_paths[0])
    if combined_df.empty:
        combined_df.to_csv(output_csv, index=False)
        return

    for path in freq_best_data_paths:
        freq_df = pd.read_csv(path)
        if freq_df.empty:
            continue

    # Merge on (Latitude, Longitude)
    merged = pd.merge(
        combined_df, freq_df,
        on=["Latitude", "Longitude"],
        suffixes=("", "_2")
    )
    # Carry whatever per-antenna columns the winner has, rather than a
    # hand-written list: a fixed list silently drops every column added
    # later, which is how RSSI and bandwidth went missing from
    # multi-frequency runs while surviving single-frequency ones.
    point_cols = [c for c in combined_df.columns if not c.startswith("Best Antenna")]
    best_cols = [c for c in combined_df.columns if c.startswith("Best Antenna")]

    final_rows = []
    for _, row in merged.iterrows():
        second_wins = (row["Best Antenna SINR (dB)_2"] > row["Best Antenna SINR (dB)"])
        out = {c: row[c] for c in point_cols}
        for c in best_cols:
            out[c] = row.get(f"{c}_2", row[c]) if second_wins else row[c]
        final_rows.append(out)
    combined_df = pd.DataFrame(final_rows)
    combined_df.to_csv(output_csv, index=False)

def combine_scenario_antennas(points_gdf):
    """
        Splits 'points_gdf' into real and virtual antennas (depending on 'type').
        """
    real_ants = []
    virtual_ants = []
    for _, row in points_gdf.iterrows():
        lat_ = row.geometry.y
        lon_ = row.geometry.x
        ptype = row.get("type", "real_antenna")
        az_ = row.get("azimuth", 0.0)
        freq_ = row.get("frequency", 3560.0)
        ptx_ = row.get("Ptx", 47.0)
        gtx_ = row.get("Gtx", 19.5)
        h_ = row.get("height", 30.0)
        dt_ = row.get("downtilt", 0.0)
        dt_ = 0.0 if pd.isna(dt_) else float(dt_)
        name_ = row.get("name", None)
        name_ = None if pd.isna(name_) else str(name_)
        bw_ = row.get("bandwidth_mhz", None)
        bw_ = 10.0 if bw_ is None or pd.isna(bw_) else float(bw_)
        if "custom" in str(ptype) or "basestation" in str(ptype) or "virtual" in str(ptype):
            virtual_ants.append({
                "lat": lat_, "lon": lon_,
                "azimuth": az_,
                "frequency": freq_,
                "Ptx": ptx_,
                "Gtx": gtx_,
                "height": h_,
                "downtilt": dt_,
                "bandwidth_mhz": bw_,
                "name": name_,
                "type": "virtual"
            })
        else:
            real_ants.append({
                "lat": lat_, "lon": lon_,
                "azimuth": az_,
                "frequency": freq_,
                "Ptx": ptx_,
                "Gtx": gtx_,
                "height": h_,
                "downtilt": dt_,
                "bandwidth_mhz": bw_,
                "name": name_,
                "type": "real"
            })
    return real_ants, virtual_ants

def generate_coverage_points(polygons_gdf, all_antennas, default_n=3000):
    from .cli_utils import _ask, _ask_int, _info, _section, _warn
    _section("Coverage Points")
    from_csv = _ask(
        "Read coverage points from CSV? [y/n]",
        default="n"
    ).lower().startswith("y")
    if from_csv:
        coverage_points = []
        with open("coverage_points.csv", "r") as file:
            reader = csv.reader(file)
            points = [
                {
                    "lat": float(row[1]),
                    "lon": float(row[0]),
                    "data": list(row[2:])
                    
                }
                for row in reader
            ]
            coverage_points = []
            with open("real_points.csv", "w") as output:
                for point in points:
                    if polygons_gdf.contains(Point(point["lon"], point["lat"])).any():
                        coverage_points.append(point)
                        output.write(f"{point['lon']},{point['lat']}")
                        for curr in point["data"]: output.write(f",{curr}")
                        output.write("\n")
            print(f"{len(coverage_points)} points read")
            return coverage_points
    
    npoints = _ask_int("Number of measurement points", default=default_n)

    if len(polygons_gdf) > 0:
        coverage_points = generate_points_in_polygon(polygons_gdf.iloc[0].geometry, npoints)
    else:
        if all_antennas:
            lat0, lon0 = all_antennas[0]["lat"], all_antennas[0]["lon"]
        else:
            lat0, lon0 = (42.90076, -78.8650)  # arbitrary fallback
            coverage_points = generate_points_in_circle(lat0, lon0, 2000, npoints)

    return coverage_points



def save_antennas_description(all_antennas, output_folder):
    """
        Saves a text file listing all antennas and their attributes.
        """
    txt_path = os.path.join(output_folder, "antennas_description.txt")
    os.makedirs(output_folder, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Antennas Description ===\n")
        f.write("Lists all antennas (real + virtual) with their parameters.\n\n")
        for i, ant in enumerate(all_antennas, start=1):
            f.write(f"Antenna #{i}\n")
            f.write(f"      Type: {ant.get('type', 'N/A')}\n")
            f.write(f"      Latitude:  {ant.get('lat','N/A')}\n")
            f.write(f"      Longitude: {ant.get('lon','N/A')}\n")
            f.write(f"      Azimuth:   {ant.get('azimuth','N/A')}\n")
            f.write(f"      Frequency (MHz): {ant.get('frequency','N/A')}\n")
            f.write(f"      Ptx (dBm):               {ant.get('Ptx','N/A')}\n")
            f.write(f"      Gtx (dBi):               {ant.get('Gtx','N/A')}\n")
            f.write(f"      Height (m):              {ant.get('height','N/A')}\n")
            f.write("----------------------------------------------\n")
            from .cli_utils import _info as _cinfo
            _cinfo(f"Antennas description saved at: {txt_path}")

def perform_coverage_analysis(all_antennas, polygons_gdf, coverage_points, path_loss_params, 
                              output_folder, point_heights, antenna_heights, diffraction_losses,
                              skip_grid_prompt=False, grid_granularity=50, 
                              title_suffix=""):
    """
    Runs multi-frequency coverage computation and plots:
    1) Identify all distinct frequency channels
    2) For each frequency, compute coverage
    3) Combine coverage data from all frequencies
    4) Generate final composite coverage plots
    """
    diffraction_losses = [0 for _ in diffraction_losses]
    from .cli_utils import _info, _ok, _ask_int, _section, _warn
    # Delete existing output folder if it exists
    if os.path.exists(output_folder):
        _info("Removing existing results directory…")
        shutil.rmtree(output_folder, ignore_errors=True)

    os.makedirs(output_folder, exist_ok=True)

    if not skip_grid_prompt:
        _section("Grid-Based SINR Heatmap")
        grid_granularity = _ask_int("Grid resolution (cells per side)", default=50)

    # Distinct frequencies
    distinct_freqs = sorted(set(ant["frequency"] for ant in all_antennas))
    freq_best_data_paths = []

    # For each frequency, do a separate coverage analysis
    for freq in distinct_freqs:
        freq_folder = os.path.join(output_folder, f"freq_{freq}_data")
        os.makedirs(freq_folder, exist_ok=True)

        # Save description for these antennas
        freq_ants = [a for a in all_antennas if a["frequency"] == freq]
        save_antennas_description(freq_ants, freq_folder)

        # Compute coverage with these freq_ants
        process_metrics(
            antennas=freq_ants,
            points=coverage_points,
            output_folder=freq_folder,
            path_loss_model=path_loss_params["model"],
            base_height=path_loss_params["base_height"],
            mobile_height=path_loss_params["mobile_height"],
            alpha=path_loss_params["alpha"],
            beta=path_loss_params["beta"],
            n=path_loss_params["n"],
            shadowing=path_loss_params["shadowing"],
            omni=path_loss_params["omni"],
            environment=path_loss_params["environment"],
            # ACI: pass all antennas (all frequencies) when enabled
            all_antennas_for_aci=(all_antennas if path_loss_params.get("enable_aci") else None),
            aclr_db=path_loss_params.get("aclr_db", 45.0),
            acs_db=path_loss_params.get("acs_db", 33.0),
            channel_bw_mhz=path_loss_params.get("channel_bw_mhz", 10.0),
            scenario=path_loss_params.get("scenario", "uma"),
            point_heights=point_heights,
            antenna_heights=antenna_heights,
            diffraction_losses=diffraction_losses
        )

        # Rename best_data.csv -> {freq}_best_data.csv
        old_best_csv = os.path.join(freq_folder, "best_data.csv")
        new_best_csv = os.path.join(freq_folder, f"{freq}_best_data.csv")
        if os.path.isfile(old_best_csv):
            if os.path.isfile(new_best_csv):
                os.remove(new_best_csv)
            os.rename(old_best_csv, new_best_csv)
            freq_best_data_paths.append(new_best_csv)

        # Separate real vs virtual (for plotting arrows/legend)
        freq_real_ants = [f for f in freq_ants if f["type"] == "real"]
        freq_virt_ants = [f for f in freq_ants if f["type"] == "virtual"]

        # Generate the plots for this frequency
        plot_best_rsrp_heatmap(
            output_folder=freq_folder,
            polygon_gdf=polygons_gdf,
            real_antennas=freq_real_ants,
            virtual_antennas=freq_virt_ants,
            title_suffix=f"_freq_{freq}",
            csv_path=new_best_csv
        )

        plot_best_sinr_heatmap(
            output_folder=freq_folder,
            polygon_gdf=polygons_gdf,
            real_antennas=freq_real_ants,
            virtual_antennas=freq_virt_ants,
            title_suffix=f"_freq_{freq}",
            csv_path=new_best_csv
        )

        plot_best_throughput_heatmap(
            output_folder=freq_folder,
            polygon_gdf=polygons_gdf,
            real_antennas=freq_real_ants,
            virtual_antennas=freq_virt_ants,
            title_suffix=f"_freq_{freq}",
            csv_path=new_best_csv
        )

        plot_grid_based_sinr_heatmap(
            output_folder=freq_folder,
            polygon_gdf=polygons_gdf,
            real_antennas=freq_real_ants,
            virtual_antennas=freq_virt_ants,
            granularity=grid_granularity,
            title_suffix=f"_freq_{freq}",
            csv_path=new_best_csv
        )

    # Combine all freq_{freq}_best_data.csv into final best_data.csv
    final_best_data_path = os.path.join(output_folder, "best_data.csv")
    combine_frequency_best_data(freq_best_data_paths, final_best_data_path)

    # Final plots using the combined best_data.csv
    _info("Generating final plots from combined best_data.csv (best across all frequencies)…")
    real_ants = [a for a in all_antennas if a["type"] == "real"]
    virt_ants = [a for a in all_antennas if a["type"] == "virtual"]

    plot_best_rsrp_heatmap(
        output_folder=output_folder,
        polygon_gdf=polygons_gdf,
        real_antennas=real_ants,
        virtual_antennas=virt_ants,
        title_suffix=f"{title_suffix}_final",
        csv_path=final_best_data_path
    )
    plot_best_sinr_heatmap(
        output_folder=output_folder,
        polygon_gdf=polygons_gdf,
        real_antennas=real_ants,
        virtual_antennas=virt_ants,
        title_suffix=f"{title_suffix}_final",
        csv_path=final_best_data_path
    )
    plot_best_throughput_heatmap(
        output_folder=output_folder,
        polygon_gdf=polygons_gdf,
        real_antennas=real_ants,
        virtual_antennas=virt_ants,
        title_suffix=f"{title_suffix}_final",
        csv_path=final_best_data_path
    )
    plot_grid_based_sinr_heatmap(
        output_folder=output_folder,
        polygon_gdf=polygons_gdf,
        real_antennas=real_ants,
        virtual_antennas=virt_ants,
        granularity=grid_granularity,
        title_suffix=f"{title_suffix}_final",
        csv_path=final_best_data_path
    )
    _ok(f"Done! Results saved in '{output_folder}'.")
